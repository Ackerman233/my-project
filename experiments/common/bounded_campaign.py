"""Frozen batches under the user's 2026-10-10 cumulative USD200 authorization.

The historical USD95 runner remains unchanged for audit/reproduction. This
module uses the same ledger, lock, reservations and spend checkpoint.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import time

from dynamic_skill.infrastructure.campaign_budget import CampaignBudget, CampaignHalt
from experiments.alfworld.common.alfworld_campaign import CAMPAIGN
from experiments.alfworld.dynamic_interrupt.interrupt_recovery_ablation import config_hash
from experiments.common.bounded_research95 import freeze, check_frozen, auth_key
from experiments.common.puzzle_pilot import save

AUTHORIZED_LIMIT_USD = 200
USER_AUTHORIZATION = '低成本检查能否模仿skillopt的思路，接入另一个judge model？或者换不同prompt构建角色但是相同api model这样；美元限额提升到200美元'


class StageBudget(CampaignBudget):
    def __init__(self, root, *, path=None):
        self.root = Path(root)
        ledger = Path(path) if path is not None else CAMPAIGN / 'budget.json'
        if not ledger.is_file():
            raise ValueError('Existing cumulative ledger required; do not create a fresh budget')
        super().__init__(ledger, deadline=0)

    def reserve(self, label, reservation_usd=None):
        state = self.read()
        auth = state.get(auth_key(self.root), {})
        if (state['limit_usd'] != AUTHORIZED_LIMIT_USD or
                auth.get('cumulative_limit_usd') != AUTHORIZED_LIMIT_USD):
            raise CampaignHalt('budget_authorization_changed')
        if not auth or auth.get('closed') or time.time() >= auth['deadline_unix']:
            raise CampaignHalt('deadline_stop')
        if len(state['requests']) >= auth['max_requests']:
            raise CampaignHalt('request_cap_stop')
        reserve = state['reservation_usd'] if reservation_usd is None else reservation_usd
        if type(reserve) not in (float, int) or not math.isfinite(reserve) or reserve <= 0:
            raise ValueError('Positive finite reservation required')
        if self.effective_spend(state) + reserve > auth['batch_spend_ceiling'] + 1e-9:
            raise CampaignHalt('budget_stop')
        if state['consecutive_errors'] >= 2:
            raise CampaignHalt('api_error_guard')
        recent = state['requests'][auth['start_request_count']:]
        if sum(r['status'] in ('pending', 'unknown_usage') for r in recent) >= auth['max_new_unknown']:
            raise CampaignHalt('unknown_cost_stop')
        return super().reserve(label, reserve)


def raise_authorized_limit(audit_directory):
    """Apply the explicit user limit increase without reopening paid requests."""
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = StageBudget(audit_directory)
        state = budget.read()
        if any(r['status'] == 'pending' for r in state['requests']):
            raise ValueError('Pending requests require settlement before changing the limit')
        if state['limit_usd'] not in (95, 105, 160, AUTHORIZED_LIMIT_USD):
            raise ValueError('Unexpected previous authorization')
        if state['limit_usd'] == AUTHORIZED_LIMIT_USD:
            return
        save(Path(audit_directory) / 'budget_before_limit_change.json', state)
        change = {'time_unix': time.time(), 'previous_limit_usd': state['limit_usd'],
            'new_limit_usd': AUTHORIZED_LIMIT_USD, 'user_authorization': USER_AUTHORIZATION,
            'request_count': len(state['requests']), 'cumulative_usd': budget.effective_spend(state)}
        state.setdefault('limit_change_history', []).append(change)
        state.update(limit_usd=AUTHORIZED_LIMIT_USD, max_requests=len(state['requests']), deadline_unix=time.time())
        budget.write(state)
        save(Path(audit_directory) / 'limit_change.json', change)


def authorize(root):
    root = Path(root)
    cfg = json.loads((root / 'config.json').read_text())
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = StageBudget(root)
        state = budget.read()
        assert cfg['cumulative_limit_usd'] == state['limit_usd'] == AUTHORIZED_LIMIT_USD
        assert not state['policy_violation'] and not any(r['status'] == 'pending' for r in state['requests'])
        assert time.time() >= state['cooldown_until'] and state['consecutive_errors'] < 2
        assert auth_key(root) not in state
        check_frozen(root, cfg)
        save(root / 'budget_before_authorization.json', state)
        auth = {'config_sha256': config_hash(cfg), 'cumulative_limit_usd': AUTHORIZED_LIMIT_USD,
            'start_request_count': len(state['requests']),
            'max_requests': len(state['requests']) + cfg['limits']['max_requests'],
            'deadline_unix': time.time() + cfg['limits'].get('seconds', 14400),
            'batch_spend_ceiling': min(AUTHORIZED_LIMIT_USD, budget.effective_spend(state) + cfg['limits']['batch_usd']),
            'max_new_unknown': cfg['limits'].get('max_new_unknown', 1),
            'initial_unknown': sum(r['status'] == 'unknown_usage' for r in state['requests']),
            'user_authorization': USER_AUTHORIZATION, 'closed': False}
        state[auth_key(root)] = auth
        state.update(max_requests=auth['max_requests'], deadline_unix=auth['deadline_unix'],
                     max_unknown_requests=auth['initial_unknown'] + auth['max_new_unknown'])
        budget.write(state)
        save(root / 'authorization.json', auth)
        print(json.dumps({'authorized': str(root), 'cumulative_limit_usd': AUTHORIZED_LIMIT_USD,
                          'batch_spend_ceiling': auth['batch_spend_ceiling']}))


def open_batch(root):
    cfg = json.loads((Path(root) / 'config.json').read_text())
    budget = StageBudget(root)
    state = budget.read()
    auth = state[auth_key(root)]
    assert not auth['closed'] and auth['config_sha256'] == config_hash(cfg)
    assert cfg['cumulative_limit_usd'] == state['limit_usd'] == auth['cumulative_limit_usd'] == AUTHORIZED_LIMIT_USD
    assert not state['policy_violation'] and not any(r['status'] == 'pending' for r in state['requests'])
    check_frozen(root, cfg)
    os.environ.setdefault('TARGET_AZURE_OPENAI_API_KEY', Path('/opt/data/private/api-key').read_text().strip())
    return cfg, budget


def batch_accounting(root):
    root = Path(root)
    budget = StageBudget(root)
    state = budget.read()
    auth = state[auth_key(root)]
    rows = [r for r in state['requests'][auth['start_request_count']:] if r['label'].startswith(str(root) + '/')]
    return {'requests': len(rows), 'returned_usage': sum(r['status'] == 'returned_usage' for r in rows),
        'unknown_usage': sum(r['status'] == 'unknown_usage' for r in rows),
        'cost_usd': sum(r['budget_charge_usd'] for r in rows),
        'actor_cost_usd': sum(r['budget_charge_usd'] for r in rows if r['label'].endswith('/actor')),
        'optimizer_cost_usd': sum(r['budget_charge_usd'] for r in rows if r['label'].endswith('/optimizer')),
        'judge_cost_usd': sum(r['budget_charge_usd'] for r in rows if r['label'].endswith('/judge')),
        'cumulative_requests': len(state['requests']), 'cumulative_usd': budget.effective_spend(state),
        'limit_usd': state['limit_usd'], 'pending': sum(r['status'] == 'pending' for r in state['requests'])}


def rearm_after_cooldown(audit_directory, health_report):
    """Explicit recovery after expired cooldown and three fresh unauthenticated probes.

    No old reservation/request is removed, no stage is reopened, and no payment
    authorization is expanded. A subsequent new frozen batch still needs authorize.
    """
    report = json.loads(Path(health_report).read_text())
    now = time.time()
    if (report.get('paid_requests') != 0 or report.get('route') not in ('direct', 'isolated_tw02') or
            not 0 <= now-report.get('finished_unix', 0) <= 600 or
            len(report.get('probes', [])) < 3 or
            any(not 200 <= r.get('status', 0) < 500 for r in report['probes'])):
        raise ValueError('Three recent successful no-key TLS/HTTP probes required')
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = StageBudget(audit_directory)
        state = budget.read()
        if now < state['cooldown_until'] or any(r['status'] == 'pending' for r in state['requests']) or state['policy_violation']:
            raise ValueError('Cooldown/pending/policy guard remains active')
        save(Path(audit_directory) / 'budget_before_error_rearm.json', state)
        state.setdefault('error_guard_recovery_history', []).append({
            'time_unix':now,'previous_consecutive_errors':state['consecutive_errors'],
            'health_report':str(Path(health_report).resolve()),
            'reason':'Expired cooldown; explicit switch to freshly probed '+report['route']+' route',
            'unknown_reservations_retained':True,'requests_reopened':False})
        state.update(consecutive_errors=0, max_requests=len(state['requests']),deadline_unix=now)
        budget.write(state)

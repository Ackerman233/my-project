"""New bounded protocol diagnostic; preserve all historical configs and outcomes."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import time

from experiments import space_boundary_trial as trial
from experiments.interrupt_recovery_ablation import read, digest, config_hash
from experiments.puzzle_pilot import Settings, save
from dynamic_skill.control_only_evolution import ControlPolicy
from dynamic_skill.campaign_budget import CampaignBudget

ROOT = Path('results/protocol_diagnostic_20261007')
AUTH = 'protocol_diagnostic_20261007_authorization'


def configure():
    trial.ROOT, trial.AUTH = ROOT, AUTH
    trial.ARMS = {'P_protocol': {'macro_mode': True, 'discovery_interrupt': False}}


def prepare():
    configure()
    old = read(Path('results/space_subskills_v2_20261003/config.json'))
    tasks = list(old['tasks'])
    previously_used = set()
    candidates = sorted(Path('/opt/data/private/data/alfworld/json_2.1.1/valid_seen').glob('*/*/game.tw-pddl'))
    for path in Path('results').rglob('config.json'):
        if path == ROOT / 'config.json':
            continue
        try:
            previous = json.dumps(read(path))
        except (ValueError, OSError):
            continue
        for game in candidates:
            if str(game) in previous:
                previously_used.add(str(game))
    for kind in ('pick_and_place_simple', 'look_at_obj_in_light'):
        game = next(p for p in candidates if p.parent.parent.name.startswith(kind + '-') and str(p) not in previously_used)
        tasks.append({'id': 'protocol-new-' + kind, 'gamefile': str(game),
                      'game_sha256': digest(game), 'benchmark': 'alfworld'})
    sources = sorted(set(trial.SOURCES + ['dynamic_skill/space_boundary.py',
        'dynamic_skill/space_subskills.py', 'experiments/space_boundary_trial.py',
        'experiments/protocol_diagnostic_trial.py', 'experiments/protocol_replay.py',
        'experiments/campaign_runner.py']))
    cfg = {'protocol': 'subskills-normalized-v3-protocol-diagnostic',
           'model': old['model'], 'tasks': tasks, 'task_skill': old['task_skill'],
           'settings': asdict(Settings(model=old['model'], max_calls=12, max_steps=50,
               max_tokens=384, temperature=0, request_timeout=45, episode_seconds=210, action_only=False)),
           'policy': asdict(ControlPolicy()), 'enable_thinking': False,
           'common': {'dynamic': True, 'history_evidence': True, 'state_check': True},
           'arms': trial.ARMS, 'source_hashes': {p: digest(Path(p)) for p in sources},
           'job_order': [[i, 'P_protocol'] for i in range(len(tasks))],
           'gate': {'format_errors': 0, 'infrastructure_errors': 0,
                    'all_episodes_accounted': True},
           'note': 'Two previously observed tasks + two deterministically selected new development tasks. '
                   'Selection uses paths and prior configs only, not solutions or outcomes. '
                   'This tests protocol compatibility, not interrupt efficacy or independent generalization.'}
    if (ROOT / 'config.json').exists():
        assert read(ROOT / 'config.json') == cfg, 'Frozen config differs; use a new namespace'
    else:
        save(ROOT / 'config.json', cfg)
    print('PREPARED', config_hash(cfg))


def authorize():
    cfg = read(ROOT / 'config.json')
    with (trial.CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(trial.CAMPAIGN / 'budget.json', deadline=0)
        data = budget.read()
        assert AUTH not in data and not data['policy_violation']
        assert not any(r['status'] == 'pending' for r in data['requests'])
        assert time.time() >= data['cooldown_until'] and data['limit_usd'] == 85
        assert all(digest(Path(p)) == h for p, h in cfg['source_hashes'].items())
        save(ROOT / 'budget_before_authorization.json', data)
        auth = {'config_sha256': config_hash(cfg), 'deadline_unix': time.time() + 3600,
                'max_requests': len(data['requests']) + 48,
                'start_request_count': len(data['requests']),
                'batch_spend_ceiling': min(data['limit_usd'], budget.effective_spend(data) + 4),
                'user_authorization': '2026-10-07 user instructed continuing concrete experiments autonomously and supplied API credential path; existing cumulative USD85 ceiling retained'}
        data[AUTH] = auth
        data.update(deadline_unix=auth['deadline_unix'], max_requests=auth['max_requests'])
        budget.write(data)
        print('AUTHORIZED', json.dumps(auth))


def advance():
    configure()
    if not os.environ.get('TARGET_AZURE_OPENAI_API_KEY'):
        # Credential file is explicitly provided by the user; never print its contents.
        os.environ['TARGET_AZURE_OPENAI_API_KEY'] = Path('/opt/data/private/api-key').read_text().strip()
    os.environ.setdefault('ALFWORLD_DATA', '/opt/data/private/data/alfworld')
    trial.advance()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'authorize', 'summary'])
    args = parser.parse_args()
    configure()
    {'prepare': prepare, 'authorize': authorize, 'summary': trial.summary}[args.action]()

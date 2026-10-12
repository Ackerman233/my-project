"""Paired post-prefix recovery with real saved queues and verified public states."""
import argparse
import copy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from dynamic_skill.campaign_budget import CampaignBudget, BudgetPlanner, CampaignHalt
from dynamic_skill.control_only_evolution import ControlExecutor, ControlPolicy
from dynamic_skill.alfworld_recovery import recovery_prompt
from dynamic_skill.micro_loop_protocol import make_prompt, parse_response
from dynamic_skill.space_subskills import INSTRUCTIONS, SYSTEM_SKILL, expand, normalize_response
from experiments.alfworld_campaign import CAMPAIGN
from experiments.interrupt_recovery_ablation import digest, read, config_hash, SOURCES
from experiments.puzzle_pilot import ApiPlanner, Settings, save

ROOT = Path('results/prefix_recovery_20261007')
AUTH = 'prefix_recovery_20261007_authorization'
ARMS = ('continue_tail', 'interrupt_replan')


def environment(gamefile):
    os.environ.setdefault('ALFWORLD_DATA', '/opt/data/private/data/alfworld')
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    return build_alfworld_env(env_num=1, eval_dataset='eval_in_distribution', seed=42,
                             is_train=False, specific_gamefiles=[gamefile])


def replay_prefix(env, case):
    path = Path(case['source_trace'])
    assert digest(path) == case['source_sha256'], 'source trace changed'
    initial, trace = read(path.parent / 'initial.json'), read(path)
    obs, _ = env.reset({})
    assert obs['anchor'][0] == initial['observation'], 'initial observation mismatch'
    prefix = copy.deepcopy(trace[:case['step'] + 1])
    assert prefix[-1]['remaining_before_interrupt'] == case['tail'], 'source tail mismatch'
    for step in prefix:
        assert list(env.envs.get_admissible_commands[0]) == step['admissible_before'], 'public commands mismatch'
        obs, rewards, dones, infos = env.step(['<think></think><action>' + step['action'] + '</action>'])
        assert obs['anchor'][0] == step['env_feedback'], 'prefix observation mismatch'
        assert bool(dones[0]) == step['done'], 'prefix terminal mismatch'
    assert not dones[0], 'prefix already terminal'
    admissible = list(env.envs.get_admissible_commands[0])
    if case['kind'] == 'surface_open_contradiction':
        assert case['tail'][0].startswith('open ') and case['tail'][0] not in admissible
        assert 'On the ' in obs['anchor'][0] and ', you see ' in obs['anchor'][0]
    elif case['kind'] == 'rejected_action_nonempty_tail':
        assert obs['anchor'][0].strip().lower() == 'nothing happens.'
    else:
        raise AssertionError('unsupported real opportunity')
    checkpoint = {'observation': obs['anchor'][0], 'admissible': admissible,
                  'task': initial['task'], 'prefix_steps': len(prefix),
                  'game_sha256': digest(Path(initial['gamefile']))}
    return initial, prefix, checkpoint


def prepare():
    audit = read(ROOT / 'search_audit.json')
    source = read(Path('results/protocol_diagnostic_v5_20261007/config.json'))
    cases = copy.deepcopy(audit['cases'])
    assert cases
    for case in cases:
        path = Path(case['source_trace'])
        initial = read(path.parent / 'initial.json')
        env = environment(initial['gamefile'])
        try:
            _, _, checkpoint = replay_prefix(env, case)
        finally:
            env.close()
        case['checkpoint'] = checkpoint
        case['checkpoint_sha256'] = config_hash(checkpoint)
        case['source_initial_sha256'] = digest(path.parent / 'initial.json')
    sources = sorted(set(SOURCES) | {'dynamic_skill/space_subskills.py', 'dynamic_skill/space_boundary.py',
        'experiments/audit_search_coverage.py', 'experiments/prefix_recovery_trial.py',
        'experiments/campaign_runner.py', 'tests/test_search_coverage_audit.py', 'tests/test_prefix_recovery_trial.py'})
    cfg = {'protocol': 'real-prefix-recovery-v1', 'model': source['model'], 'task_skill': source['task_skill'],
           'settings': asdict(Settings(model=source['model'], max_calls=4, max_steps=12, max_tokens=384,
               temperature=0, request_timeout=45, episode_seconds=210, action_only=False)),
           'policy': source['policy'], 'cases': cases, 'arms': ARMS,
           'source_hashes': {p: digest(Path(p)) for p in sources},
           'search_audit_sha256': digest(ROOT / 'search_audit.json'),
           'job_order': [[case['id'], arm] for i, case in enumerate(cases)
                         for arm in (ARMS if i % 2 == 0 else tuple(reversed(ARMS)))],
           'note': 'Same verified real prefix and original model tail in each pair. '
               'Continue executes the original tail without interruption; interrupt discards it before replanning. '
               'Both then use the same dynamic controller, prompt and 4-new-call/12-new-action budget. '
               'Prefix actions/calls are sunk costs and excluded from suffix costs. One run per arm; development diagnostic only.'}
    if (ROOT / 'config.json').exists():
        assert read(ROOT / 'config.json') == cfg, 'never change a frozen trial'
    else:
        save(ROOT / 'config.json', cfg)
        for path in sources:
            dst = ROOT / 'source_snapshot' / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    print(json.dumps({'prepared_cases': len(cases), 'prefixes_verified': len(cases), 'paid_calls': 0,
                      'config_sha256': config_hash(cfg)}))


class PrefixBudget(CampaignBudget):
    def reserve(self, label, reservation_usd=None):
        data = self.read()
        auth = data[AUTH]
        reserve = data['reservation_usd'] if reservation_usd is None else reservation_usd
        if time.time() >= auth['deadline_unix']:
            raise CampaignHalt('deadline_stop')
        if len(data['requests']) >= auth['max_requests']:
            raise CampaignHalt('request_cap_stop')
        if self.effective_spend(data) + reserve > auth['batch_spend_ceiling']:
            raise CampaignHalt('budget_stop')
        return super().reserve(label, reservation_usd)


def authorize():
    cfg = read(ROOT / 'config.json')
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = PrefixBudget(CAMPAIGN / 'budget.json', deadline=0)
        data = budget.read()
        assert AUTH not in data and data['limit_usd'] == 85 and not data['policy_violation']
        assert not any(r['status'] == 'pending' for r in data['requests'])
        assert time.time() >= data['cooldown_until']
        assert all(digest(Path(p)) == sha for p, sha in cfg['source_hashes'].items())
        save(ROOT / 'budget_before_authorization.json', data)
        auth = {'config_sha256': config_hash(cfg), 'start_request_count': len(data['requests']),
                'max_requests': len(data['requests']) + len(cfg['job_order']) * cfg['settings']['max_calls'],
                'deadline_unix': time.time() + 3600,
                'batch_spend_ceiling': min(85, budget.effective_spend(data) + 8),
                'user_authorization': '2026-10-07 user explicitly requested failure coverage analysis, real interrupt opportunities and fixed-prefix recovery comparison; cumulative USD85 retained'}
        data[AUTH] = auth
        data.update(deadline_unix=auth['deadline_unix'], max_requests=auth['max_requests'])
        budget.write(data)
        print('AUTHORIZED', json.dumps(auth))


def run_suffix(env, case, arm, cfg, planner, out):
    initial, history, checkpoint = replay_prefix(env, case)
    assert config_hash(checkpoint) == case['checkpoint_sha256']
    save(out / 'checkpoint.json', checkpoint)
    save(out / 'prefix.json', history)
    policy = ControlPolicy(**cfg['policy'])
    ctl = ControlExecutor(policy, dynamic=True, state_check=True)
    ctl.start(initial['observation'], initial['task'])
    ctl.action_history = [r['action'] for r in history]
    ctl.observation_history = [initial['observation']] + [r['env_feedback'] for r in history]
    for i, step in enumerate(history):
        ctl.monitor.check(step['action'], step['env_feedback'], [], i,
                          {'admissible': step['admissible_before']})
    original_tail = arm == 'continue_tail'
    if original_tail:
        ctl.dynamic = False
        ctl.load_chunk(case['tail'])
    else:
        ctl.last_replan_reason = case['reason'] + '; publicly observed queue/state contradiction, not an injected environment failure'
    observation = checkpoint['observation']
    calls, suffix = [], []
    settings = Settings(**cfg['settings'])
    status = 'step_limit'
    started = time.perf_counter()
    for local_step in range(settings.max_steps):
        if time.perf_counter() - started >= settings.episode_seconds:
            status = 'wall_time_limit'
            break
        admissible = list(env.envs.get_admissible_commands[0])
        if ctl.needs_plan:
            ctl.dynamic = True
            if len(calls) >= settings.max_calls:
                status = 'model_call_limit'
                break
            limit = min(policy.max_actions, settings.max_steps - len(suffix))
            context = {'task': initial['task'], 'initial': initial['observation'],
                       'observation': observation, 'admissible': admissible,
                       'trace': history, 'anomaly': ctl.last_replan_reason}
            prompt = make_prompt(cfg['task_skill'], limit, 'Replan from the current observation.')('', **context) + INSTRUCTIONS
            if context['anomaly']:
                prompt = recovery_prompt(prompt, **context, output_protocol='thought_actions')
            prompt += f'\nPost-prefix budget remaining: {settings.max_calls-len(calls)} model calls and {settings.max_steps-len(suffix)} environment actions.'
            try:
                planner.before_call()
            except CampaignHalt as halt:
                status = halt.status
                break
            call = {'call_id': len(calls) + 1, 'suffix_step': len(suffix), 'prompt': prompt,
                    'system_prompt': SYSTEM_SKILL, 'actions': [], 'action_limit': limit, 'status': 'pending'}
            calls.append(call)
            save(out / 'model_calls.json', calls)
            try:
                call.update(planner(prompt))
                usage = call.get('usage')
                if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) and usage[k] >= 0
                    for k in ('prompt_tokens', 'completion_tokens')):
                    status = call['status'] = 'unknown_cost_stop'
                    break
                details = (call.get('usage') or {}).get('completion_tokens_details') or {}
                if call.get('reasoning_content') or details.get('reasoning_tokens', 0):
                    status = call['status'] = 'thinking_policy_stop'
                    break
                text, normalized = normalize_response(call['response'])
                actions, reason = parse_response(text, limit)
                primitive, provenance = expand(actions, context, limit)
                call.update(status='ok', protocol_normalized=normalized, model_native_actions=actions,
                            actions=primitive, reasoning=reason, subskill_expansion=provenance)
                ctl.load_chunk(primitive)
            except ValueError as exc:
                status = call['status'] = 'format_error'
                call['parse_error'] = str(exc)
                break
            except Exception as exc:
                status = call['status'] = 'api_error'
                call['error_type'] = type(exc).__name__
                break
            finally:
                save(out / 'model_calls.json', calls)
            print(case['id'], arm, 'call', len(calls), 'suffix_steps', len(suffix), call['status'], flush=True)
        action = ctl.next_action()
        is_original_tail = original_tail
        ctl.before_action(action, observation, admissible)
        obs, rewards, dones, infos = env.step(['<think></think><action>' + action + '</action>'])
        observation = obs['anchor'][0]
        ctl.after_action(list(env.envs.get_admissible_commands[0]))
        event = ctl.observe(action, observation, reward=float(rewards[0]), done=bool(dones[0]), info=infos[0])
        row = {'step': len(history), 'suffix_step': len(suffix), 'action': action,
               'admissible_before': admissible, 'valid': action in admissible,
               'env_feedback': observation, 'reward': float(rewards[0]), 'done': bool(dones[0]),
               'won': bool(infos[0].get('won', False)), 'model_call_id': len(calls),
               'original_tail_action': is_original_tail, 'checker_status': event.check.status.value,
               'reasoning': calls[-1].get('reasoning', '') if calls else '',
               **ctl.last_execution_audit}
        history.append(row)
        suffix.append(row)
        save(out / 'suffix_trace.json', suffix)
        if original_tail and not ctl.pending_actions:
            original_tail = False
            ctl.dynamic = True
            if event.check.should_interrupt:
                ctl.last_replan_reason = event.check.reason
        if dones[0]:
            status = 'success' if infos[0].get('won', False) or rewards[0] > 0 else 'environment_failure'
            break
        if event.decision.kind.value == 'stop_failure':
            status = 'interrupt_budget_exhausted'
            break
    result = {'case_id': case['id'], 'arm': arm, 'status': status, 'success': status == 'success',
              'checkpoint_sha256': config_hash(checkpoint), 'prefix_steps': len(read(out / 'prefix.json')),
              'model_calls': len(calls), 'suffix_steps': len(suffix),
              'invalid_actions': sum(r['checker_status'] == 'invalid_action' for r in suffix),
              'initial_discarded_actions': case['tail'] if arm == 'interrupt_replan' else [],
              'subsequent_interrupt_count': ctl.interruption_count,
              'source_kind': case['kind'], 'original_tail_executed': sum(r['original_tail_action'] for r in suffix)}
    save(out / 'result.json', result)
    return result


def summary():
    cfg = read(ROOT / 'config.json')
    rows = [read(p) for p in ROOT.glob('online/*/*/result.json')]
    pairs = []
    for case in cfg['cases']:
        group = {r['arm']: r for r in rows if r['case_id'] == case['id']}
        pairs.append({'case_id': case['id'], 'kind': case['kind'], 'paired': len(group) == 2,
                      'checkpoint_identical': len(group) == 2 and len({r['checkpoint_sha256'] for r in group.values()}) == 1,
                      'arms': group})
    result = {'attempted': len(rows), 'scheduled': len(cfg['job_order']), 'pairs': pairs,
              'arms': {arm: {'attempted': sum(r['arm'] == arm for r in rows),
                  'successes': sum(r['success'] for r in rows if r['arm'] == arm),
                  **{k: sum(r.get(k, 0) for r in rows if r['arm'] == arm) for k in
                     ['model_calls', 'suffix_steps', 'invalid_actions', 'cost_usd']}} for arm in ARMS},
              'note': cfg['note']}
    save(ROOT / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['pairs', 'note']}))


def advance():
    os.environ.setdefault('TARGET_AZURE_OPENAI_API_KEY', Path('/opt/data/private/api-key').read_text().strip())
    cfg = read(ROOT / 'config.json')
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = PrefixBudget(CAMPAIGN / 'budget.json', deadline=0)
        data = budget.read()
        auth = data[AUTH]
        assert auth['config_sha256'] == config_hash(cfg) and data['limit_usd'] == 85
        assert time.time() < auth['deadline_unix'] and not data['policy_violation']
        assert not any(r['status'] == 'pending' for r in data['requests'])
        assert not any(r['status'] != 'returned_usage' or r.get('error_type') for r in data['requests'][auth['start_request_count']:])
        assert digest(ROOT / 'search_audit.json') == cfg['search_audit_sha256']
        assert all(digest(Path(p)) == sha for p, sha in cfg['source_hashes'].items())
        for case_id, arm in cfg['job_order']:
            case = next(c for c in cfg['cases'] if c['id'] == case_id)
            out = ROOT / 'online' / case_id / arm
            if out.exists():
                assert (out / 'result.json').exists(), 'incomplete prior suffix; audit required'
                assert read(out / 'result.json')['status'] in ('success', 'model_call_limit', 'step_limit',
                    'wall_time_limit', 'format_error', 'environment_failure', 'interrupt_budget_exhausted')
                continue
            path = Path(case['source_trace'])
            assert digest(path.parent / 'initial.json') == case['source_initial_sha256']
            gamefile = read(path.parent / 'initial.json')['gamefile']
            assert digest(Path(gamefile)) == case['checkpoint']['game_sha256']
            env = environment(gamefile)
            api = ApiPlanner(Settings(**cfg['settings']), system_prompt=SYSTEM_SKILL)
            before = budget.effective_spend(data)
            try:
                result = run_suffix(env, case, arm, cfg, BudgetPlanner(api, budget, str(out)), out)
            finally:
                api.close()
                env.close()
            result['cost_usd'] = budget.effective_spend(budget.read()) - before
            save(out / 'result.json', result)
            summary()
            return
        summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'authorize', 'summary'])
    args = parser.parse_args()
    {'prepare': prepare, 'authorize': authorize, 'summary': summary}[args.action]()

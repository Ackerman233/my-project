"""Frozen rules, unseen physical tasks, real model queues, three-arm prefix trial."""
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
from dynamic_skill.control_only_evolution import ControlExecutor, ControlPolicy, run_control_episode
from dynamic_skill.alfworld_recovery import recovery_prompt
from dynamic_skill.micro_loop_protocol import make_prompt, parse_response
from dynamic_skill.space_subskills import INSTRUCTIONS, SYSTEM_SKILL, expand, normalize_response
from dynamic_skill.queue_revision import revise_head, Truth
from dynamic_skill.queue_evidence import AlfworldEvidence
from experiments.alfworld_campaign import CAMPAIGN
from experiments.interrupt_recovery_ablation import digest, read, config_hash, SOURCES
from experiments.puzzle_pilot import ApiPlanner, Settings, save
from experiments.prefix_recovery_trial import environment

ROOT = Path('results/queue_revision_transfer_20261007')
AUTH = 'queue_revision_transfer_20261007_authorization'
ARMS = ('continue_tail', 'interrupt_replan', 'evidence_repair')
RULES = ('dynamic_skill/queue_revision.py', 'dynamic_skill/queue_evidence.py', 'dynamic_skill/alfworld_open_guard.py')
HALTS = {'api_error', 'unknown_cost_stop', 'thinking_policy_stop', 'budget_stop', 'deadline_stop', 'request_cap_stop', 'api_cooldown'}


def select_tasks():
    candidates = sorted(Path('/opt/data/private/data/alfworld/json_2.1.1/valid_seen').glob('*/*/game.tw-pddl'))
    # Exclude entire task-scene definitions, not just a game's random instance.
    used = set()
    for path in Path('results').rglob('*.json'):
        if ROOT in path.parents or 'source_snapshot' in path.parts:
            continue
        if path.name not in ('config.json', 'initial.json', 'tasks.json'):
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        for game in candidates:
            if str(game) in text:
                used.add(game.parent.parent.name)
    selected = []
    for family in ('pick_and_place_simple', 'look_at_obj_in_light'):
        objects = set()
        for game in candidates:
            name = game.parent.parent.name
            if not name.startswith(family + '-') or name in used:
                continue
            object_type = name.split('-')[1]
            if object_type in objects:
                continue
            objects.add(object_type)
            used.add(name)
            selected.append({'id': 'transfer-' + family + '-' + str(len(objects)),
                'gamefile': str(game), 'game_sha256': digest(game), 'benchmark': 'alfworld',
                'family': family, 'object_type': object_type})
            if len(objects) == 3:
                break
        assert len(objects) == 3
    return selected


def prepare():
    assert not (ROOT / 'config.json').exists(), 'Frozen config already exists; do not overwrite'
    old = read(Path('results/protocol_diagnostic_v5_20261007/config.json'))
    sources = sorted(set(SOURCES) | set(RULES) | {
        'dynamic_skill/space_subskills.py', 'dynamic_skill/space_boundary.py',
        'experiments/prefix_recovery_trial.py', 'experiments/queue_revision_transfer_trial.py',
        'tests/test_queue_revision_principles.py', 'tests/test_queue_revision_transfer_trial.py'})
    cfg = {'protocol': 'frozen-queue-revision-unseen-physical-task-v1',
        'model': old['model'], 'task_skill': old['task_skill'], 'policy': old['policy'],
        'tasks': select_tasks(), 'arms': ARMS,
        'collection_settings': asdict(Settings(model=old['model'], max_calls=12, max_steps=50,
            max_tokens=384, temperature=0, request_timeout=45, episode_seconds=210, action_only=False)),
        'settings': asdict(Settings(model=old['model'], max_calls=5, max_steps=16,
            max_tokens=384, temperature=0, request_timeout=45, episode_seconds=210, action_only=False)),
        'source_hashes': {p: digest(Path(p)) for p in sources},
        'rule_hashes': {p: digest(Path(p)) for p in RULES},
        'selector': 'First nonterminal trace step with nonempty original model/expanded tail whose head is FALSE under frozen public incomplete-list evidence. Otherwise first nonterminal nonempty tail as untriggered control. No task replacement. Prefix success/failure is not a selection criterion.',
        'intervention': 'Only the original checkpoint queue differs. Continue consumes it unchanged; clear discards it; repair checks current evidence and dependency before each next original-tail action. Afterwards all arms use identical dynamic controller, prompt, model and budgets. No new search policy.',
        'limits': {'batch_usd': 15, 'max_requests': 162, 'seconds': 7200},
        'note': 'Six new task-scene definitions, same two supported families and same ALFWorld split. Tests held-out-task transfer, not held-out-family/environment transfer. One suffix run per arm; gateway nondeterminism possible. Prefix collection costs excluded from suffix comparisons but included in ledger.'}
    save(ROOT / 'config.json', cfg)
    for p in sources:
        dst = ROOT / 'source_snapshot' / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    print('PREPARED', config_hash(cfg), [(t['id'], t['object_type']) for t in cfg['tasks']])


class TransferBudget(CampaignBudget):
    def reserve(self, label, reservation_usd=None):
        data = self.read()
        auth = data[AUTH]
        reservation = data['reservation_usd'] if reservation_usd is None else reservation_usd
        if time.time() >= auth['deadline_unix']:
            raise CampaignHalt('deadline_stop')
        if len(data['requests']) >= auth['max_requests']:
            raise CampaignHalt('request_cap_stop')
        if self.effective_spend(data) + reservation > auth['batch_spend_ceiling']:
            raise CampaignHalt('budget_stop')
        return super().reserve(label, reservation_usd)


def authorize():
    cfg = read(ROOT / 'config.json')
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = TransferBudget(CAMPAIGN / 'budget.json', deadline=0)
        data = budget.read()
        assert AUTH not in data and data['limit_usd'] == 85 and not data['policy_violation']
        assert not any(r['status'] == 'pending' for r in data['requests'])
        assert time.time() >= data['cooldown_until']
        assert all(digest(Path(p)) == sha for p, sha in cfg['source_hashes'].items())
        save(ROOT / 'budget_before_authorization.json', data)
        auth = {'config_sha256': config_hash(cfg), 'start_request_count': len(data['requests']),
            'max_requests': len(data['requests']) + cfg['limits']['max_requests'],
            'deadline_unix': time.time() + cfg['limits']['seconds'],
            'batch_spend_ceiling': min(85, budget.effective_spend(data) + cfg['limits']['batch_usd']),
            'user_authorization': '2026-10-07 explicit request to freeze rules and run new-task three-arm prefix recovery comparison; cumulative USD85 retained'}
        data[AUTH] = auth
        data.update(deadline_unix=auth['deadline_unix'], max_requests=auth['max_requests'])
        budget.write(data)
        print('AUTHORIZED', json.dumps(auth))


def replay_prefix(env, case):
    path = Path(case['source_trace'])
    assert digest(path) == case['source_sha256'], 'source trace changed'
    assert digest(path.parent / 'initial.json') == case['source_initial_sha256']
    initial, trace = read(path.parent / 'initial.json'), read(path)
    obs, _ = env.reset({})
    assert obs['anchor'][0] == initial['observation'], 'initial observation mismatch'
    history = copy.deepcopy(trace[:case['step'] + 1])
    assert history[-1]['remaining_before_interrupt'] == case['tail']
    assert digest(Path(initial['gamefile'])) == case['game_sha256']
    for row in history:
        assert list(env.envs.get_admissible_commands[0]) == row['admissible_before']
        obs, _, dones, _ = env.step(['<think></think><action>' + row['action'] + '</action>'])
        assert obs['anchor'][0] == row['env_feedback']
        assert bool(dones[0]) == row['done']
    assert not dones[0]
    checkpoint = {'observation': obs['anchor'][0], 'admissible': list(env.envs.get_admissible_commands[0]),
        'task': initial['task'], 'prefix_steps': len(history), 'game_sha256': digest(Path(initial['gamefile']))}
    return initial, history, checkpoint


def select_case(task, path):
    trace = read(path)
    eligible = [r for r in trace if not r['done'] and r.get('remaining_before_interrupt')]
    if not eligible:
        return None
    selected = eligible[0]
    # The next row's admissible_before is public post-action evidence. The last
    # unexecuted state must be replayed to acquire its current command list.
    env = environment(task['gamefile'])
    try:
        obs, _ = env.reset({})
        evidence = {}
        for row in trace:
            obs, _, _, _ = env.step(['<think></think><action>' + row['action'] + '</action>'])
            evidence[row['step']] = list(env.envs.get_admissible_commands[0])
        for row in eligible:
            proof = AlfworldEvidence(row['env_feedback'], evidence[row['step']], complete=False).executable(row['remaining_before_interrupt'][0])
            if proof.value is Truth.FALSE:
                selected = row
                break
        head_proof = AlfworldEvidence(selected['env_feedback'], evidence[selected['step']], complete=False).executable(selected['remaining_before_interrupt'][0])
    finally:
        env.close()
    case = {'id': task['id'], 'task_id': task['id'], 'source_trace': str(path),
        'source_sha256': digest(path), 'source_initial_sha256': digest(path.parent / 'initial.json'),
        'game_sha256': task['game_sha256'], 'step': selected['step'],
        'tail': selected['remaining_before_interrupt'],
        'kind': 'falsified_head' if head_proof.value is Truth.FALSE else 'untriggered_control',
        'head_verdict': head_proof.value.value,
        'reason': 'Frozen original queue review: ' + head_proof.reason}
    env = environment(task['gamefile'])
    try:
        _, _, checkpoint = replay_prefix(env, case)
    finally:
        env.close()
    case['checkpoint'], case['checkpoint_sha256'] = checkpoint, config_hash(checkpoint)
    return case


def freeze_cases():
    cfg = read(ROOT / 'config.json')
    cases, omissions = [], []
    for task in cfg['tasks']:
        directory = ROOT / 'collection' / 'predictions' / (cfg['model'] + '__' + task['id'] + '__fixed_chunk')
        assert (directory / 'result.json').exists(), 'collection incomplete'
        assert read(directory / 'result.json')['status'] not in HALTS
        case = select_case(task, directory / 'conversation.json') if (directory / 'conversation.json').exists() else None
        if case:
            cases.append(case)
        else:
            omissions.append({'task_id': task['id'], 'reason': 'No nonterminal nonempty original queue; no replacement'})
    frozen = {'config_sha256': config_hash(cfg), 'cases': cases, 'omissions': omissions,
        'job_order': [[case['id'], arm] for i, case in enumerate(cases)
            for arm in ARMS[i % 3:] + ARMS[:i % 3]]}
    if (ROOT / 'cases.json').exists():
        assert read(ROOT / 'cases.json') == json.loads(json.dumps(frozen))
    else:
        save(ROOT / 'cases.json', frozen)
    print('FROZEN_CASES', len(cases), 'falsified', sum(c['kind'] == 'falsified_head' for c in cases), 'omissions', len(omissions))

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
    original_tail = arm != 'interrupt_replan'
    if original_tail:
        ctl.dynamic = False
        ctl.load_chunk(case['tail'])
    else:
        ctl.last_replan_reason = case['reason'] + '; publicly observed queue/state contradiction, not an injected environment failure'
    observation = checkpoint['observation']
    calls, suffix, revisions = [], [], []
    repair_active = arm == 'evidence_repair'
    settings = Settings(**cfg['settings'])
    status = 'step_limit'
    started = time.perf_counter()
    for local_step in range(settings.max_steps):
        if time.perf_counter() - started >= settings.episode_seconds:
            status = 'wall_time_limit'
            break
        admissible = list(env.envs.get_admissible_commands[0])
        if repair_active and ctl.pending_actions:
            revision = revise_head(list(ctl.pending_actions), AlfworldEvidence(observation, admissible, complete=False))
            revisions.append({'suffix_step': len(suffix), 'revision': asdict(revision)})
            save(out / 'queue_revisions.json', revisions)
            if revision.decision == 'skip_head':
                assert ctl.pending_actions.popleft() == revision.skipped[0]
            elif revision.decision == 'replan':
                discarded = list(ctl.pending_actions)
                ctl.pending_actions.clear()
                revisions[-1]['discarded_for_replan'] = discarded
                save(out / 'queue_revisions.json', revisions)
                ctl.last_replan_reason = case['reason'] + '; publicly observed queue/state contradiction, not an injected environment failure'
                ctl.dynamic = True
                original_tail = repair_active = False
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
            original_tail = repair_active = False
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
              'source_kind': case['kind'], 'original_tail_executed': sum(r['original_tail_action'] for r in suffix),
              'format_errors': sum(c['status'] == 'format_error' for c in calls),
              'queue_revision_count': len(revisions),
              'skipped_falsified_heads': sum(len(r['revision']['skipped']) for r in revisions),
              'repair_initial_decision': revisions[0]['revision']['decision'] if revisions else None,
              'revision_replan_discarded': sum(len(r.get('discarded_for_replan', [])) for r in revisions)}
    save(out / 'result.json', result)
    return result



def summary():
    cfg = read(ROOT / 'config.json')
    rows = [read(p) for p in ROOT.glob('online/*/*/result.json')]
    frozen = read(ROOT / 'cases.json') if (ROOT / 'cases.json').exists() else {'cases': [], 'job_order': [], 'omissions': []}
    report = {'attempted': len(rows), 'scheduled': len(frozen['job_order']),
        'prefixes': len(frozen['cases']), 'falsified_prefixes': sum(c['kind'] == 'falsified_head' for c in frozen['cases']),
        'omissions': frozen['omissions'], 'arms': {arm: {'attempted': sum(r['arm'] == arm for r in rows),
            'successes': sum(r['success'] for r in rows if r['arm'] == arm),
            **{k: sum(r.get(k, 0) for r in rows if r['arm'] == arm) for k in
                ('model_calls', 'suffix_steps', 'invalid_actions', 'format_errors', 'cost_usd', 'skipped_falsified_heads')}} for arm in ARMS},
        'pairs': [{'case_id': case['id'], 'kind': case['kind'],
            'complete': len([r for r in rows if r['case_id'] == case['id']]) == 3,
            'checkpoint_identical': len({r['checkpoint_sha256'] for r in rows if r['case_id'] == case['id']}) == 1,
            'arms': {r['arm']: r for r in rows if r['case_id'] == case['id']}} for case in frozen['cases']],
        'note': cfg['note']}
    save(ROOT / 'summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs', 'note')}))


def advance():
    os.environ.setdefault('TARGET_AZURE_OPENAI_API_KEY', Path('/opt/data/private/api-key').read_text().strip())
    cfg = read(ROOT / 'config.json')
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = TransferBudget(CAMPAIGN / 'budget.json', deadline=0)
        data = budget.read()
        auth = data[AUTH]
        assert auth['config_sha256'] == config_hash(cfg) and data['limit_usd'] == 85
        assert time.time() < auth['deadline_unix'] and not data['policy_violation']
        assert not any(r['status'] == 'pending' for r in data['requests'])
        assert not any(r['status'] != 'returned_usage' or r.get('error_type') for r in data['requests'][auth['start_request_count']:])
        assert all(digest(Path(p)) == sha for p, sha in cfg['source_hashes'].items())
        for task in cfg['tasks']:
            directory = ROOT / 'collection' / 'predictions' / (cfg['model'] + '__' + task['id'] + '__fixed_chunk')
            if directory.exists():
                assert (directory / 'result.json').exists(), 'incomplete collection; audit required'
                assert read(directory / 'result.json')['status'] not in HALTS
                continue
            assert digest(Path(task['gamefile'])) == task['game_sha256']
            env = environment(task['gamefile'])
            api = ApiPlanner(Settings(**cfg['collection_settings']), system_prompt=SYSTEM_SKILL)
            before = budget.effective_spend(data)
            try:
                result = run_control_episode(env, task, Settings(**cfg['collection_settings']), ROOT / 'collection',
                    BudgetPlanner(api, budget, str(directory)), cfg['task_skill'], ControlPolicy(**cfg['policy']),
                    dynamic=False, history_evidence=True, state_check=False, macro_mode=True)
            finally:
                api.close()
                env.close()
            result['cost_usd'] = budget.effective_spend(budget.read()) - before
            save(directory / 'result.json', result)
            print('COLLECTED', task['id'], result['status'], flush=True)
            return
        if not (ROOT / 'cases.json').exists():
            freeze_cases()
            frozen = read(ROOT / 'cases.json')
            auth['cases_sha256'] = digest(ROOT / 'cases.json')
            data[AUTH] = auth
            budget.write(data)
        assert digest(ROOT / 'cases.json') == auth['cases_sha256']
        frozen = read(ROOT / 'cases.json')
        assert frozen['config_sha256'] == config_hash(cfg)
        for case_id, arm in frozen['job_order']:
            out = ROOT / 'online' / case_id / arm
            if out.exists():
                assert (out / 'result.json').exists(), 'incomplete suffix; audit required'
                assert read(out / 'result.json')['status'] not in HALTS
                continue
            case = next(c for c in frozen['cases'] if c['id'] == case_id)
            env = environment(read(Path(case['source_trace']).parent / 'initial.json')['gamefile'])
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
    parser.add_argument('action', choices=['prepare', 'authorize', 'advance', 'summary'])
    args = parser.parse_args()
    {'prepare': prepare, 'authorize': authorize, 'advance': advance, 'summary': summary}[args.action]()

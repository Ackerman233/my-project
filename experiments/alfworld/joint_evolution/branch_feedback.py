"""Training-only intervention at real, replay-verified nonterminal queues.

One checkpoint per task, chosen before suffix outcomes: first contradiction or
observed target discovery, otherwise first ordinary queue. Both arms share the
public prefix, primitive operator interface, skill, controller and suffix budget.
Continuing commits the saved tail; interrupting discards it. After that initial
intervention both run the same controller. This is a noisy two-rollout signal,
not an oracle label or an estimate for every individual policy parameter.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from dynamic_skill.controllers.control_only_evolution import ControlExecutor
from dynamic_skill.infrastructure.campaign_budget import CampaignHalt
from dynamic_skill.alfworld.micro_loop_protocol import make_prompt, parse_response
from dynamic_skill.alfworld.alfworld_recovery import recovery_prompt
from experiments.common.puzzle_pilot import save
from .engine import CLEAN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_case(row, directory):
    eligible = [(i, s) for i, s in enumerate(row['trace'])
                if not s['done'] and s.get('remaining_before_interrupt')]
    if not eligible:
        return None
    flagged = [(i, s) for i, s in eligible if not s.get('valid', True) or
               s.get('state_check', {}).get('blocks_tail') or
               s.get('discovery_check', {}).get('triggered')]
    index, step = (flagged or eligible)[0]
    trace_path = Path(directory) / 'predictions' / row['id'] / 'conversation.json'
    return {'id': row['task_id'], 'step': index,
        'tail': step['remaining_before_interrupt'], 'source_trace': str(trace_path),
        'source_sha256': digest(trace_path), 'initial_sha256': digest(trace_path.parent / 'initial.json'),
        'kind': 'public_contradiction_or_discovery' if flagged else 'ordinary_nonempty_queue',
        'reason': step.get('interruption_reason') or 'Training intervention at a real nonempty action queue.'}


def replay(env, case, genome):
    path = Path(case['source_trace'])
    assert digest(path) == case['source_sha256'] and digest(path.parent / 'initial.json') == case['initial_sha256']
    initial = json.loads((path.parent / 'initial.json').read_text())
    prefix = json.loads(path.read_text())[:case['step'] + 1]
    obs, _ = env.reset({})
    assert obs['anchor'][0] == initial['observation'], 'Initial observation mismatch'
    ctl = ControlExecutor(genome.policy.executable(), dynamic=True,
                          state_check=genome.policy.state_check, discovery_interrupt=genome.policy.discovery_interrupt)
    ctl.start(initial['observation'], initial['task'])
    for step in prefix:
        admissible = list(env.envs.get_admissible_commands[0])
        assert admissible == step['admissible_before'], 'Prefix commands mismatch'
        if ctl.needs_plan:
            ctl.load_chunk(step['original_action_queue'])
        action = ctl.next_action()
        assert action == step['action'], 'Prefix queue mismatch'
        ctl.before_action(action, obs['anchor'][0], admissible)
        obs, rewards, dones, infos = env.step(['<think></think><action>' + action + '</action>'])
        assert obs['anchor'][0] == step['env_feedback'] and bool(dones[0]) == step['done'], 'Prefix replay mismatch'
        ctl.after_action(list(env.envs.get_admissible_commands[0]))
        ctl.observe(action, obs['anchor'][0], reward=float(rewards[0]), done=bool(dones[0]), info=infos[0])
        assert ctl.last_execution_audit['remaining_before_interrupt'] == step['remaining_before_interrupt']
    assert prefix and not prefix[-1]['done'] and prefix[-1]['remaining_before_interrupt'] == case['tail']
    checkpoint = {'observation': obs['anchor'][0], 'admissible': list(env.envs.get_admissible_commands[0]),
                  'task': initial['task'], 'prefix_steps': len(prefix), 'game_sha256': digest(initial['gamefile'])}
    return initial, prefix, checkpoint, ctl


def suffix(env, case, arm, genome, settings, planner, out):
    initial, history, checkpoint, ctl = replay(env, case, genome)
    save(out / 'checkpoint.json', checkpoint)
    save(out / 'prefix.json', history)
    ctl.pending_actions.clear()
    # Use the exact saved tail, not an imagined replacement. Loading it does not
    # grant a fresh planning allowance or alter the actual environment prefix.
    original_tail = arm == 'continue_tail'
    ctl.recovery_remaining = 0
    if original_tail:
        ctl.dynamic = False
        ctl.load_chunk(case['tail'])
        ctl.last_replan_reason = ''
    else:
        ctl.last_replan_reason = case['reason']
    observation = checkpoint['observation']
    calls, trace = [], []
    status, started = 'step_limit', time.perf_counter()
    for _ in range(settings.max_steps):
        if time.perf_counter() - started >= settings.episode_seconds:
            status = 'wall_time_limit'
            break
        admissible = list(env.envs.get_admissible_commands[0])
        if ctl.needs_plan:
            ctl.dynamic = True
            if len(calls) >= settings.max_calls:
                status = 'model_call_limit'
                break
            limit = min(ctl.planning_limit, settings.max_steps - len(trace))
            context = {'task': initial['task'], 'initial': initial['observation'],
                       'observation': observation, 'admissible': admissible,
                       'trace': history, 'anomaly': ctl.last_replan_reason}
            prompt = make_prompt(genome.skill_text, limit, 'Replan from the current observation.')('', **context)
            if genome.policy.history_evidence and context['anomaly']:
                prompt = recovery_prompt(prompt, **context, output_protocol='thought_actions')
            prompt += f'\nPost-prefix budget remaining: {settings.max_calls-len(calls)} model calls and {settings.max_steps-len(trace)} environment actions.'
            call = {'call_id': len(calls)+1, 'prompt': prompt, 'status': 'pending'}
            try:
                planner.before_call()
                calls.append(call)
                save(out / 'model_calls.json', calls)
                call.update(planner(prompt))
                actions, reason = parse_response(call['response'], limit)
                call.update(status='ok', actions=actions, reasoning=reason)
                ctl.load_chunk(actions)
            except CampaignHalt as exc:
                status = call['status'] = exc.status
                break
            except ValueError as exc:
                status = call['status'] = 'format_error'
                call['parse_error'] = str(exc)
                break
            finally:
                save(out / 'model_calls.json', calls)
        action = ctl.next_action()
        was_original = original_tail
        ctl.before_action(action, observation, admissible)
        obs, rewards, dones, infos = env.step(['<think></think><action>' + action + '</action>'])
        observation = obs['anchor'][0]
        ctl.after_action(list(env.envs.get_admissible_commands[0]))
        event = ctl.observe(action, observation, reward=float(rewards[0]), done=bool(dones[0]), info=infos[0])
        row = {'step': len(history), 'action': action, 'env_feedback': observation,
               'admissible_before': admissible, 'valid': action in admissible, 'done': bool(dones[0]),
               'original_tail_action': was_original, 'reasoning': calls[-1].get('reasoning', '') if calls else '',
               'checker_status': event.check.status.value, **ctl.last_execution_audit}
        history.append(row)
        trace.append(row)
        save(out / 'suffix_trace.json', trace)
        if original_tail and not ctl.pending_actions:
            original_tail, ctl.dynamic = False, True
            if event.check.should_interrupt:
                ctl.last_replan_reason = event.check.reason
        if dones[0]:
            status = 'success' if infos[0].get('won', False) or rewards[0] > 0 else 'environment_failure'
            break
        if event.decision.kind.value == 'stop_failure':
            status = 'interrupt_budget_exhausted'
            break
    result = {'case_id': case['id'], 'arm': arm, 'status': status, 'success': status == 'success',
              'checkpoint_sha256': hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest(),
              'model_calls': len(calls), 'environment_steps': len(trace),
              'invalid_actions': sum(not s['valid'] for s in trace), 'cost_usd': planner.cost,
              'original_tail_executed': sum(s['original_tail_action'] for s in trace)}
    save(out / 'result.json', result)
    return result


def compare(case, rows):
    a, b = rows['continue_tail'], rows['interrupt_replan']
    valid = a['status'] in CLEAN and b['status'] in CLEAN and a['checkpoint_sha256'] == b['checkpoint_sha256']
    label = 'unknown'
    if valid and a['success'] != b['success']:
        label = 'interrupt_win' if b['success'] else 'continue_win'
    elif valid and a['success'] and b['success']:
        metrics = ('model_calls', 'environment_steps', 'invalid_actions', 'cost_usd')
        if all(b[k] <= a[k] for k in metrics) and any(b[k] < a[k] for k in metrics):
            label = 'interrupt_efficiency_win'
        elif all(a[k] <= b[k] for k in metrics) and any(a[k] < b[k] for k in metrics):
            label = 'continue_efficiency_win'
        else:
            label = 'successful_tradeoff_or_tie'
    return {'valid_pair': valid, 'label': label, 'kind': case['kind'], 'prefix_steps': case['step']+1,
            'original_tail': case['tail'], 'arms': rows,
            'caveat': 'One rollout per branch; noisy training evidence, not a population causal effect. Both failed => unknown.'}


def enrich(transport, rows, evidence, genome, directory):
    cases = []
    by_id = {e['task_id']: e for e in evidence}
    for row in rows:
        if row['task_id'] in by_id:
            case = checkpoint_case(row, directory.parent / 'execution_feedback')
            if case:
                cases.append(case)
    save(directory / 'cases.json', cases)  # frozen before any suffix outcome
    settings = replace(transport.settings, max_calls=4, max_steps=16)
    for index, case in enumerate(cases):
        initial = json.loads((Path(case['source_trace']).parent / 'initial.json').read_text())
        results = {}
        arms = ('continue_tail', 'interrupt_replan')
        for arm in arms if index % 2 == 0 else reversed(arms):
            out = directory / case['id'] / arm
            from .resume import reuse_branch
            cached=reuse_branch(transport.branch_cache,case,arm,out)
            if cached is not None:
                results[arm]=cached
                continue
            env = transport.environment(initial['gamefile'])
            actor = transport.actor(str(out / 'actor'))
            try:
                result = suffix(env, case, arm, genome, settings, actor, out)
            finally:
                env.close()
            result['cost_usd'] = sum(r['budget_charge_usd'] for r in transport.budget.read()['requests']
                                      if r['label'] == str(out / 'actor'))
            save(out / 'result.json', result)
            results[arm] = result
            print('branch', case['id'], arm, result['status'], flush=True)
            if result['status'] not in CLEAN | {'format_error'}:
                raise CampaignHalt(result['status'])
        by_id[case['id']]['branch_comparison'] = compare(case, results)
        save(directory / case['id'] / 'comparison.json', by_id[case['id']]['branch_comparison'])
    return evidence

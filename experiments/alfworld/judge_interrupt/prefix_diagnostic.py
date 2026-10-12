"""Replay every first judge/rule disagreement, with both suffix choices.

Development diagnostic after the frozen episode trial, never a held-out test.
One disagreement per task/role, selected before buying any suffix outcome.
"""
import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import time

from dynamic_skill.infrastructure.campaign_budget import CampaignHalt
from experiments.alfworld.common.alfworld_campaign import CAMPAIGN
from experiments.alfworld.joint_evolution.branch_feedback import suffix, compare, digest
from experiments.alfworld.joint_evolution.engine import CLEAN
from experiments.alfworld.joint_evolution.schema import Genome
from experiments.alfworld.joint_evolution.runtime import LiveTransport
from experiments.common.bounded_campaign import (AUTHORIZED_LIMIT_USD, freeze, authorize,
    open_batch, batch_accounting, auth_key)
from experiments.common.puzzle_pilot import Settings, save


def prepare(source, root):
    completion = json.loads((source / 'completion.json').read_text())
    if completion['status'] != 'complete':
        raise ValueError('Complete the frozen trial before choosing diagnostic cases')
    cfg = json.loads((source / 'config.json').read_text())
    rows = json.loads((source / 'run/results.json').read_text())
    cases, inputs = [], {str(source / 'config.json'): digest(source / 'config.json'),
                         str(source / 'run/results.json'): digest(source / 'run/results.json')}
    for row in sorted(rows, key=lambda r: (r['split'], r['task_id'], r['arm'])):
        if row['arm'] == 'rules':
            continue
        out = source / 'run' / row['split'] / row['task_id'] / row['arm']
        path = out / 'predictions' / row['id'] / 'conversation.json'
        if not path.exists():
            continue
        for index, step in enumerate(json.loads(path.read_text())):
            check = step.get('judge', {})
            if not check.get('changed_rule_decision'):
                continue
            assert step['remaining_before_interrupt'] and not step['done']
            callpath = out / 'judge_calls.json'
            call = json.loads(callpath.read_text())[check['call_id'] - 1]
            case = {'id': row['task_id'] + '__' + row['arm'], 'step': index,
                'tail': step['remaining_before_interrupt'], 'source_trace': str(path),
                'source_sha256': digest(path), 'initial_sha256': digest(path.parent / 'initial.json'),
                'kind': 'first_observed_judge_rule_disagreement',
                'reason': step.get('interruption_reason') or check['rule_reason'],
                'judge_decision': check['decision'], 'judge_evidence': check.get('evidence'),
                'historical_judge_cost_usd': call['cost_usd'],
                'judge_head_was_admissible': step['remaining_before_interrupt'][0] in call['context']['admissible_now']}
            cases.append(case)
            for p in (path, path.parent / 'initial.json', callpath):
                inputs[str(p)] = digest(p)
            break
    if not cases:
        raise ValueError('No genuine decision disagreement to diagnose')
    paths = sorted([*Path('dynamic_skill').rglob('*.py'), *Path('experiments').rglob('*.py'),
                    *Path('skillopt/skillopt').rglob('*.py')])
    new = {'protocol': 'judge-disagreement-prefix-v1', 'cumulative_limit_usd': AUTHORIZED_LIMIT_USD,
        'source': str(source), 'cases': cases, 'initial': cfg['initial'],
        'settings': {**cfg['settings'], 'max_calls': 4, 'max_steps': 16},
        'network_route': cfg['network_route'], 'pooled_transport': cfg['pooled_transport'],
        'limits': {'batch_usd': min(12, 3*len(cases)), 'max_requests': 8*len(cases),
                   'max_new_unknown': 1, 'seconds': 3600},
        'input_hashes': {**cfg['input_hashes'], **inputs},
        'source_hashes': {str(p): digest(p) for p in paths},
        'preregistration': 'All first rule/judge disagreements, one per task and role, regardless of episode success. '
            'Replay exact public prefix then continue original tail versus discard and replan; same original '
            'rule controller thereafter, 4 actor calls/16 actions per arm. Alternate arm order. No further judge '
            'calls or oracle state. Both failed => unknown. One suffix rollout/arm; noisy development diagnostic. '
            'Report branch metrics and an additional comparison including the already-paid judge call/cost '
            'on its chosen branch. Historical judge fee is not charged again. Infrastructure errors stop batch.'}
    freeze(root, new)


def run(root):
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, budget = open_batch(root)
        if (root / 'run').exists():
            raise ValueError('Cannot overwrite a paid diagnostic')
        (root / 'run').mkdir()
        transport = LiveTransport(root, budget, Settings(**cfg['settings']),
            pooled_transport=cfg['pooled_transport'], network_route=cfg['network_route'])
        genome = Genome.from_dict(cfg['initial'])
        results, completion = [], {'status': 'running'}
        try:
            for index, case in enumerate(cfg['cases']):
                initial = json.loads((Path(case['source_trace']).parent / 'initial.json').read_text())
                arms = ('continue_tail', 'interrupt_replan')
                values = {}
                for arm in arms if index % 2 == 0 else reversed(arms):
                    out = root / 'run' / case['id'] / arm
                    actor = transport.actor(str(out / 'actor'))
                    env = transport.environment(initial['gamefile'])
                    try:
                        value = suffix(env, case, arm, genome, transport.settings, actor, out)
                    finally:
                        env.close()
                    values[arm] = value
                    print(case['id'], arm, value['status'], value['model_calls'], flush=True)
                    if value['status'] not in CLEAN | {'format_error'}:
                        raise CampaignHalt(value['status'])
                comparison = compare(case, values)
                chosen = 'continue_tail' if case['judge_decision'] == 'continue' else 'interrupt_replan'
                net = json.loads(json.dumps(values))
                net[chosen]['cost_usd'] += case['historical_judge_cost_usd']
                net[chosen]['model_calls'] += 1
                row = {'case': case, 'suffix_comparison': comparison,
                       'with_judge_overhead': compare(case, net), 'judge_chosen_arm': chosen}
                results.append(row)
                save(root / 'run/comparisons.json', results)
            completion['status'] = 'complete'
        except CampaignHalt as exc:
            completion.update(status='stopped', reason=exc.status)
        except Exception as exc:
            completion.update(status='error', error_type=type(exc).__name__)
            raise
        finally:
            transport.close()
            save(root / 'completion.json', {**completion, 'paired_cases': len(results)})
            save(root / 'accounting.json', batch_accounting(root))
            state = budget.read()
            if not any(r['status'] == 'pending' for r in state['requests']):
                state[auth_key(root)]['closed'] = True
                state['max_requests'] = len(state['requests'])
                state['deadline_unix'] = time.time()
                budget.write(state)
            save(root / 'budget_after.json', budget.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'authorize', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(args.source.resolve(), root)
    elif args.action == 'authorize':
        authorize(root)
    else:
        run(root)

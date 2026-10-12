"""Zero-API replay of observed false-continue judgments against the hard guard.

This verifies local rejection only; it cannot establish end-to-end success gain.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynamic_skill.controllers.judge import HeadGuardChecker
from dynamic_skill.controllers.control_only_evolution import ControlPolicy, PolicyChecker
from experiments.alfworld.joint_evolution.branch_feedback import replay, digest
from experiments.alfworld.joint_evolution.schema import Genome
from experiments.alfworld.joint_evolution.runtime import LiveTransport
from experiments.common.puzzle_pilot import save


def verify(root):
    cfg = json.loads((root / 'config.json').read_text())
    genome = Genome.from_dict(cfg['initial'])
    records = []
    for path in sorted((root / 'run').rglob('judge_calls.json')):
        calls = json.loads(path.read_text())
        tracepath = next((path.parent / 'predictions').glob('*/conversation.json'))
        trace = json.loads(tracepath.read_text())
        for call in calls:
            context = call['context']
            if call.get('decision') != 'continue' or context['remaining_queue'][0] in context['admissible_now']:
                continue
            index = next(i for i, step in enumerate(trace) if step.get('judge', {}).get('call_id') == call['call_id'])
            case = {'source_trace': str(tracepath), 'source_sha256': digest(tracepath),
                    'initial_sha256': digest(tracepath.parent / 'initial.json'),
                    'step': index, 'tail': context['remaining_queue']}
            initial = json.loads((tracepath.parent / 'initial.json').read_text())
            env = LiveTransport.environment(initial['gamefile'])
            try:
                _, history, checkpoint, _ = replay(env, case, genome)
                assert checkpoint['admissible'] == context['admissible_now']
                owner = SimpleNamespace(pending_actions=deque(case['tail']), admissible_after=checkpoint['admissible'])
                checker = HeadGuardChecker(PolicyChecker(ControlPolicy()), owner)
                result = checker.check(history[-1]['action'], checkpoint['observation'],
                    [s['action'] for s in history[:-1]],
                    [initial['observation']] + [s['env_feedback'] for s in history[:-1]])
                assert result.should_interrupt and 'public_next_action_inadmissible' in result.signals
                obs, _, _, _ = env.step(['<think></think><action>' + case['tail'][0] + '</action>'])
                feedback = obs['anchor'][0]
                assert index + 1 < len(trace) and feedback == trace[index + 1]['env_feedback']
                assert 'nothing happens' in feedback.lower()
                records.append({'source_trace': str(tracepath), 'step': index, 'head': case['tail'][0],
                    'old_judge_decision': 'continue', 'new_hard_guard_decision': 'interrupt',
                    'exact_public_prefix_verified': True, 'actual_head_feedback': feedback, 'api_calls': 0})
            finally:
                env.close()
    return {'cases': records, 'verified_count': len(records), 'api_calls': 0,
            'claim': 'Known rejected NEXT commands are vetoed without a judge call; no full-episode gain claim.'}


if __name__ == '__main__':
    import httpx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    def deny(*a, **kw):
        raise AssertionError('API requests prohibited in offline replay')
    with patch.object(httpx.HTTPTransport, 'handle_request', deny), patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', deny):
        result = verify(args.root.resolve())
    save(args.out, result)
    print(json.dumps({'verified_count': result['verified_count'], 'api_calls': 0}))

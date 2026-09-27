import json
from experiments.puzzle_pilot import Settings, run_episode


def run(tmp_path, variant, chunks, mode='dynamic_interrupt'):
    answers = iter(chunks)
    def planner(prompt):
        return {'response': '<actions>' + json.dumps(next(answers)) + '</actions>',
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
    task = {'id': 'controlled', 'benchmark': 'rush_hour', 'minimum_actions': 1,
            'board': '......' '......' 'AA....' '......' '......' '......'}
    result = run_episode(task, mode, Settings(controller_variant=variant, action_only=True), tmp_path, planner)
    trace = json.loads((tmp_path / 'predictions' / ('controlled__' + mode) / 'conversation.json').read_text())
    return result, trace


def test_filter_executes_original_subsequence_without_model_call(tmp_path):
    result, trace = run(tmp_path, 'noop_tail_v1', [['A-1', 'A-2', 'A+1', 'A+3']])
    assert result['success'] and result['model_calls'] == 1
    assert [r['source_action_index'] for r in trace] == [0, 2, 3]
    assert [r['model_called'] for r in trace] == [True, False, False]
    assert result['filter_checks'] == 3 and result['filter_events'] == 1
    assert result['interrupt_count'] == 0
    assert trace[0]['checker_status'] == 'invalid_action'
    assert trace[0]['tail_filter']['skipped'] == [[0, 'illegal_noop']]


def test_strict_default_keeps_replanning_semantics(tmp_path):
    result, trace = run(tmp_path, 'strict', [['A-1', 'A+4'], ['A+4']])
    assert result['success'] and result['model_calls'] == 2
    assert result['interrupt_count'] == 1 and result['filter_events'] == 0
    assert trace[0]['discarded_actions'] == ['A+4']


def test_cycle_filter_falls_back_to_next_model_call(tmp_path):
    result, trace = run(tmp_path, 'noop_tail_v1', [['A-1', 'A+1', 'A-1', 'A+4'], ['A+4']])
    assert result['success'] and result['model_calls'] == 2
    assert trace[0]['tail_filter']['reason'] == 'cycle_replan'
    assert trace[0]['controller_decision'] == 'interrupt_and_replan'


def test_fixed_mode_does_not_enable_filter(tmp_path):
    result, trace = run(tmp_path, 'noop_tail_v1', [['A-1', 'A+4']], mode='fixed_chunk')
    assert result['success'] and result['model_calls'] == 1 and result['filter_events'] == 0

from experiments.audit_search_coverage import target, analyze_episode
from experiments.puzzle_pilot import save


def test_observed_task_grammars():
    assert target('look at pillow under the desklamp.') == 'pillow'
    assert target('examine the cd with the desklamp.') == 'cd'
    assert target('put some egg on microwave.') == 'egg'
    assert target('clean something') is None


def test_content_coverage_and_real_tail_opportunity(tmp_path):
    save(tmp_path / 'initial.json', {'task': 'put some box on dresser.'})
    save(tmp_path / 'result.json', {'task_id': 'synthetic', 'status': 'model_call_limit'})
    trace = [
        {'action': 'go to drawer 1', 'admissible_before': ['go to drawer 1', 'go to drawer 2'],
         'env_feedback': 'You arrive at drawer 1. On the drawer 1, you see nothing.',
         'remaining_before_interrupt': ['open drawer 1']},
        {'action': 'open drawer 1', 'admissible_before': ['examine drawer 1', 'go to drawer 2'],
         'env_feedback': 'Nothing happens.', 'remaining_before_interrupt': []},
        {'action': 'go to drawer 2', 'admissible_before': ['go to drawer 2'],
         'env_feedback': 'You arrive at drawer 2. The drawer 2 is closed.',
         'remaining_before_interrupt': []},
    ]
    save(tmp_path / 'conversation.json', trace)
    result = analyze_episode(tmp_path / 'conversation.json')
    assert result['content_observed_locations'] == ['drawer 1']
    assert result['content_not_observed_locations'] == ['drawer 2']
    assert result['opportunities'][0]['kind'] == 'surface_open_contradiction'
    assert result['opportunities'][0]['tail'] == ['open drawer 1']

from experiments.audit_discarded_tails import inspect_tail
from dynamic_skill.puzzles import make_environment


def env():
    return make_environment({'benchmark': 'rush_hour', 'board': '......' '......' 'AA....' '......' '......' '......'})


def test_invalid_prefix_can_have_legal_goal_tail_without_mutating_source():
    source = env()
    report = inspect_tail(source, ['A-1', 'A+4', 'A-2'])
    assert report['legal_actions'] == ['A+4']
    assert report['invalid_actions'] == ['A-1']
    assert report['counterfactual_tail_wins']
    assert not report['first_remaining_legal']
    assert not source.solved


def test_legality_not_equivalent_to_goal_progress():
    report = inspect_tail(env(), ['A+1', 'A-1'])
    assert report['first_remaining_legal']
    assert len(report['legal_actions']) == 2
    assert not report['counterfactual_tail_wins']


def test_all_invalid_and_empty_tails():
    assert inspect_tail(env(), ['A-1', 'Z+1'])['legal_actions'] == []
    assert inspect_tail(env(), [])['steps'] == []

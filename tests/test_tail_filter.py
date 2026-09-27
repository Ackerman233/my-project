from dynamic_skill.puzzles import RushHourEnv, make_environment
from dynamic_skill.tail_filter import filter_tail


def env(board=None):
    return make_environment({'benchmark': 'rush_hour', 'board': board or ('......' '......' 'AA....' '......' '......' '......')})


def scan(source, actions, **kw):
    return filter_tail(source, actions, failed_valid=False, before_failed=source.render(), **kw)


def test_skip_invalid_keep_order_stop_at_goal_and_isolate():
    source = env()
    before = source.render()
    result = scan(source, ['A-1', 'A+1', 'A+3', 'A-1'])
    assert result.decision == 'continue'
    assert result.retained == ('A+1', 'A+3') and result.retained_indices == (1, 2)
    assert result.checks == 3 and result.skipped == ((0, 'illegal_noop'),)
    assert source.render() == before and not source.solved


def test_dependencies_rechecked_after_skipping():
    source = env('......' '..B...' 'AAB...' '......' '......' '......')
    result = scan(source, ['B+5', 'A+4', 'B-1', 'A+4'])
    assert result.retained == ('B-1', 'A+4')
    assert result.retained_indices == (2, 3) and result.checks == 4


def test_cycle_discards_even_previously_legal_prefix():
    result = scan(env(), ['A+1', 'A-1', 'A+4'])
    assert result.reason == 'cycle_replan' and not result.retained
    assert result.checks == 2


def test_recent_cycle_and_scan_budget():
    source = env()
    other = env()
    other.step('A+1')
    assert scan(source, ['A+1'], recent_states=[other.render()]).reason == 'cycle_replan'
    assert scan(source, ['A+1'] * 5).checks == 0
    assert scan(source, []).reason == 'no_legal_tail'


def test_noop_precondition_required_and_unknown_env_rejected():
    source = env()
    assert filter_tail(source, ['A+4'], failed_valid=True, before_failed=source.render()).decision == 'replan'
    assert filter_tail(source, ['A+4'], failed_valid=False, before_failed='different').checks == 0
    assert filter_tail(object(), [], failed_valid=False, before_failed='').reason == 'unsupported_simulator'


def test_invalid_side_effect_reverts_all_and_preserves_live_env(monkeypatch):
    original_step = RushHourEnv.step
    def broken_step(self, action):
        result = original_step(self, 'A+1')
        result.valid = False
        return result
    monkeypatch.setattr(RushHourEnv, 'step', broken_step)
    source = env()
    before = source.render()
    result = scan(source, ['A-1'])
    assert result.reason == 'invalid_action_side_effect' and not result.retained
    assert source.render() == before


def test_sudoku_immutable_clue_and_dependency():
    from dynamic_skill.puzzles import SudokuEnv
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    source = SudokuEnv(grid)
    result = scan(source, ['clear(1,1)', 'set(1,2,2)', 'set(1,3,2)', 'set(1,3,3)'])
    assert result.retained == ('set(1,2,2)', 'set(1,3,3)')
    assert result.retained_indices == (1, 3)
    assert source.board == grid
    assert not source.solved

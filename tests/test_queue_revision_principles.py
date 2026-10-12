import pytest
from dynamic_skill.queue_revision import revise_head, Truth, Proof
from dynamic_skill.queue_evidence import AlfworldEvidence, RushHourEvidence, SudokuEvidence
from dynamic_skill.puzzles import RushHourEnv, SudokuEnv


def test_incomplete_negative_list_cannot_falsify():
    adapter = AlfworldEvidence('You are at desk 1.', ['go to shelf 1'])
    decision = revise_head(['open drawer 1', 'go to shelf 1'], adapter)
    assert decision.decision == 'replan' and decision.skipped == ()
    assert decision.proofs[0].value is Truth.UNKNOWN


def test_stale_and_conflicting_evidence_refuse_skip():
    action = ['open drawer 1', 'go to shelf 1']
    stale = AlfworldEvidence('On the drawer 1, you see nothing.', ['go to shelf 1'], observed_id='old observation')
    conflict = AlfworldEvidence('On the drawer 1, you see nothing.', ['open drawer 1', 'go to shelf 1'])
    for adapter in (stale, conflict):
        assert revise_head(action, adapter).skipped == ()
        assert revise_head(action, adapter).decision == 'replan'


@pytest.mark.parametrize('location,destination', [('drawer 1', 'shelf 1'), ('cubby 9', 'rack 8'), ('fridge 2', 'desk 7')])
def test_location_renaming_and_next_only_scope(location, destination):
    adapter = AlfworldEvidence('On the ' + location + ', you see nothing.', ['go to ' + destination])
    actions = ['open ' + location, 'go to ' + destination, 'take egg 1 from ' + location]
    result = revise_head(actions, adapter)
    assert result.decision == 'skip_head'
    assert result.pending == tuple(actions[1:])
    assert result.next_action == actions[1]  # Never certifies the later take.


def test_open_take_dependency_and_already_satisfied_effect():
    blocked = AlfworldEvidence('The drawer 1 is closed.', ['go to shelf 1'], complete=True)
    assert revise_head(['open drawer 1', 'take egg 1 from drawer 1'], blocked).next_action is None
    already_open = AlfworldEvidence('The drawer 1 is open.', ['take egg 1 from drawer 1'], complete=True)
    assert revise_head(['open drawer 1', 'take egg 1 from drawer 1'], already_open).decision == 'skip_head'


def test_failed_pickup_cannot_enable_delivery():
    adapter = AlfworldEvidence('You are not carrying anything.', ['go to desk 1'], complete=True)
    result = revise_head(['take egg 1 from fridge 1', 'move egg 1 to desk 1'], adapter)
    assert result.next_action is None and result.decision == 'replan'
    # Redundant acquisition can be skipped only with fresh evidence that delivery is permitted.
    held = AlfworldEvidence('You are carrying: a egg 1.', ['move egg 1 to desk 1'], complete=True)
    assert revise_head(['take egg 1 from fridge 1', 'move egg 1 to desk 1'], held).decision == 'skip_head'


def test_unknown_effect_contract_is_not_assumed_independent():
    adapter = AlfworldEvidence('The drawer 1 is open.', ['use desklamp 1'], complete=True)
    result = revise_head(['open drawer 1', 'use desklamp 1'], adapter)
    assert result.decision == 'replan'
    assert result.proofs[1].value is Truth.UNKNOWN


def rush_adapter():
    env = RushHourEnv('......' '..B...' 'AAB...' '......' '...CC.' '......')
    return env, RushHourEvidence(env.render())


def test_relative_rush_move_dependency_even_when_next_is_legal():
    env, adapter = rush_adapter()
    assert not env.step('B+5').valid
    assert env.step('B-1').valid
    result = revise_head(['B+5', 'B-1'], adapter)
    assert result.decision == 'replan' and result.next_action is None


def test_rush_independent_move_retained_but_blocked_path_rejected():
    env, adapter = rush_adapter()
    before = env.render()
    good = revise_head(['B+5', 'C-1'], adapter)
    assert good.decision == 'skip_head' and good.next_action == 'C-1'
    bad = revise_head(['B+5', 'A+4'], adapter)
    assert bad.next_action is None
    assert env.render() == before


def test_sudoku_clear_dependency_and_redundant_clear():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    env = SudokuEnv(grid)
    adapter = SudokuEvidence(env.render())
    assert revise_head(['clear(1,1)', 'set(1,2,1)'], adapter).next_action is None
    independent = revise_head(['clear(1,1)', 'set(1,2,2)'], adapter)
    assert independent.next_action == 'set(1,2,2)'
    redundant = revise_head(['clear(2,2)', 'set(2,2,2)'], adapter)
    assert redundant.decision == 'skip_head'


def test_sudoku_public_user_entry_markers_and_same_cell_effect():
    env = SudokuEnv([[0] * 9 for _ in range(9)])
    assert env.step('set(1,1,2)').valid
    adapter = SudokuEvidence(env.render())
    assert revise_head(['clear(1,1)'], adapter).decision == 'execute'
    assert revise_head(['set(1,1,3)', 'clear(1,1)'], adapter).decision == 'replan'


def test_no_hidden_solver_is_used(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('hidden solution access')
    monkeypatch.setattr(RushHourEnv, 'shortest_solution', forbidden)
    monkeypatch.setattr('dynamic_skill.puzzles.sudoku_solutions', forbidden)
    _, rush = rush_adapter()
    assert revise_head(['B+5', 'C-1'], rush).next_action == 'C-1'
    sudoku = SudokuEvidence(SudokuEnv([[0] * 9 for _ in range(9)]).render())
    assert revise_head(['clear(1,1)', 'set(1,1,9)'], sudoku).next_action == 'set(1,1,9)'


def test_non_authoritative_dependency_cannot_override_head_proof():
    class Adapter:
        snapshot_id = 'current'
        def executable(self, action):
            return Proof(Truth.FALSE if action == 'bad' else Truth.TRUE, 'current', 'public', True, 'fixture')
        def independent_after_skip(self, *args):
            return Proof(Truth.TRUE, 'current', 'guess', False, 'not established')
    assert revise_head(['bad', 'good'], Adapter()).next_action is None

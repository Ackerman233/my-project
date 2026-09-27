"""Deterministic puzzle environments. Runtime feedback never uses a hidden solution."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re

from .checker import CheckResult, CheckStatus, ObservationChecker


@dataclass
class PuzzleStep:
    observation: str
    valid: bool
    reason: str
    won: bool


class PuzzleChecker(ObservationChecker):
    """Visible legality and short state cycles only; no solver or solution access."""

    def check(self, action, observation, action_history=(), observation_history=(),
              reward=0.0, done=False, info=None):
        info = info or {}
        if done:
            return CheckResult(CheckStatus.SUCCESS if info.get("won") else CheckStatus.ENV_FAILURE,
                               "task completed" if info.get("won") else "terminal failure")
        if info.get("valid") is False:
            return CheckResult(CheckStatus.INVALID_ACTION, info.get("reason", "invalid action"), True)
        if observation in observation_history[-4:]:
            return CheckResult(CheckStatus.STALLED, "returned to a recently visited board", True)
        return CheckResult(CheckStatus.PROGRESS, "legal move changed the board")


def candidates(board, row, col):
    used = set(board[row]) | {board[r][col] for r in range(9)}
    used |= {board[r][c] for r in range(row // 3 * 3, row // 3 * 3 + 3)
             for c in range(col // 3 * 3, col // 3 * 3 + 3)}
    return set(range(1, 10)) - used


def sudoku_solutions(puzzle, limit=2):
    """Offline MRV backtracking for dataset validation, never called by step/checker."""
    board = [row[:] for row in puzzle]
    found = []

    def search():
        if len(found) >= limit:
            return
        empty = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
        if not empty:
            found.append([row[:] for row in board])
            return
        r, c = min(empty, key=lambda rc: len(candidates(board, *rc)))
        for value in sorted(candidates(board, r, c)):
            board[r][c] = value
            search()
            board[r][c] = 0
            if len(found) >= limit:
                break
    search()
    return found


class SudokuEnv:
    rules = (
        "Solve the 9x9 Sudoku. Each row, column and 3x3 box must contain 1..9 exactly once. "
        "Rows and columns are numbered 1..9. One action is set(r,c,v), e.g. set(1,2,7). "
        "Only empty cells can be filled; given clues cannot change. To undo your own entry, "
        "use clear(r,c), then set a replacement. Each set/clear costs one environment action. "
        "Entries marked with '*' were filled by you and can be cleared; unmarked digits are immutable clues. "
        "Only local row/column/box conflicts are rejected; accepted entries may still be globally wrong."
    )

    def __init__(self, puzzle):
        if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
            raise ValueError("expected a 9x9 board")
        self.board = [row[:] for row in puzzle]
        self.givens = {(r, c) for r in range(9) for c in range(9) if puzzle[r][c]}
        for r, c in self.givens:
            value = self.board[r][c]
            self.board[r][c] = 0
            if value not in candidates(self.board, r, c):
                raise ValueError("conflicting initial clues")
            self.board[r][c] = value

    @property
    def solved(self):
        return all(self.board[r][c] for r in range(9) for c in range(9))

    def render(self):
        return "Columns: 1 2 3 4 5 6 7 8 9\n" + "\n".join(
            f"r{r+1}: " + " ".join((str(v) + ("*" if (r, c) not in self.givens else "")) if v else "."
                                    for c, v in enumerate(row)) for r, row in enumerate(self.board))

    def step(self, action):
        set_match = re.fullmatch(r"set\(\s*([1-9])\s*,\s*([1-9])\s*,\s*([1-9])\s*\)", action.strip())
        clear_match = re.fullmatch(r"clear\(\s*([1-9])\s*,\s*([1-9])\s*\)", action.strip())
        valid, reason = False, "invalid syntax: use set(r,c,v) or clear(r,c)"
        if set_match or clear_match:
            groups = (set_match or clear_match).groups()
            r, c = int(groups[0]) - 1, int(groups[1]) - 1
            if (r, c) in self.givens:
                reason = "cannot change a given clue"
            elif clear_match:
                valid = bool(self.board[r][c])
                reason = "cleared entry" if valid else "cell is already empty"
                self.board[r][c] = 0
            elif self.board[r][c]:
                reason = "cell is occupied; clear your entry before replacing it"
            elif int(groups[2]) not in candidates(self.board, r, c):
                reason = "row, column or box conflict"
            else:
                self.board[r][c] = int(groups[2])
                valid, reason = True, "entry accepted by local constraints"
        return PuzzleStep(self.render(), valid, reason, self.solved)


class RushHourEnv:
    rules = (
        "Solve the 6x6 Rush Hour board. '.' is empty and 'x' is a fixed wall. "
        "Each capital letter is one vehicle. A is the horizontal target car in row 3; "
        "win by moving A to columns 5-6 at the right edge. Vehicles cannot rotate, overlap, "
        "jump obstacles or leave the board. One action is ID+N or ID-N, e.g. B+2. "
        "Positive means right for horizontal vehicles or down for vertical vehicles; negative means left/up. "
        "A legal slide of any positive distance counts as ONE action, not N actions. "
        "Plan the necessary rearrangements, not only moves that immediately advance A."
    )

    def __init__(self, board):
        board = board.replace("o", ".")
        if len(board) != 36:
            raise ValueError("expected a 36-cell board")
        if any(ch not in ".xABCDEFGHIJKLMNOPQRSTUVWXYZ" for ch in board):
            raise ValueError("unknown board symbol")
        self.walls = {i for i, v in enumerate(board) if v == "x"}
        self.ids = sorted(set(board) - {".", "x"})
        self.vehicles, positions = [], []
        for car in self.ids:
            cells = [i for i, v in enumerate(board) if v == car]
            if len(cells) not in (2, 3):
                raise ValueError("vehicle length must be 2 or 3")
            horizontal = len({i // 6 for i in cells}) == 1
            stride = 1 if horizontal else 6
            if cells != list(range(cells[0], cells[0] + stride * len(cells), stride)):
                raise ValueError("vehicle cells must form a contiguous line")
            fixed = cells[0] // 6 if horizontal else cells[0] % 6
            positions.append(cells[0] % 6 if horizontal else cells[0] // 6)
            self.vehicles.append((horizontal, fixed, len(cells)))
        self.state = tuple(positions)
        if "A" not in self.ids or self.vehicles[self.ids.index("A")] != (True, 2, 2):
            raise ValueError("target A must be a horizontal length-two car in row 3")

    def board_string(self, state=None):
        state = self.state if state is None else state
        board = ["x" if i in self.walls else "." for i in range(36)]
        for car, (horizontal, fixed, length), pos in zip(self.ids, self.vehicles, state):
            for offset in range(length):
                index = fixed * 6 + pos + offset if horizontal else (pos + offset) * 6 + fixed
                if board[index] != ".":
                    raise ValueError("overlapping vehicles")
                board[index] = car
        return "".join(board)

    def neighbors(self, state):
        board = self.board_string(state)
        for i, (horizontal, fixed, length) in enumerate(self.vehicles):
            pos = state[i]
            for sign in (-1, 1):
                for distance in range(1, 6):
                    edge = pos - distance if sign < 0 else pos + length - 1 + distance
                    if not 0 <= edge < 6:
                        break
                    index = fixed * 6 + edge if horizontal else edge * 6 + fixed
                    if board[index] != ".":
                        break
                    next_state = list(state)
                    next_state[i] += sign * distance
                    yield f"{self.ids[i]}{sign * distance:+d}", tuple(next_state)

    @property
    def solved(self):
        return self.state[self.ids.index("A")] == 4

    def render(self):
        board = self.board_string()
        lines = [f"r{r+1}: " + " ".join(board[r*6:r*6+6]) for r in range(6)]
        return "Columns: 1 2 3 4 5 6\n" + "\n".join(lines)

    def step(self, action):
        match = re.fullmatch(r"([A-Z])([+-])([1-5])", action.strip())
        action = action.strip()
        neighbors = dict(self.neighbors(self.state))
        valid = bool(match) and action in neighbors
        if valid:
            self.state = neighbors[action]
        reason = "vehicle moved" if valid else "illegal slide: check vehicle axis, path occupancy and board bounds"
        return PuzzleStep(self.render(), valid, reason, self.solved)

    def shortest_solution(self, max_states=200000):
        """Offline BFS with all legal slide distances; not exposed to planner/checker."""
        start = self.state
        queue = deque([start])
        parents = {start: None}
        target_index = self.ids.index("A")
        while queue:
            state = queue.popleft()
            if state[target_index] == 4:
                path = []
                while parents[state] is not None:
                    previous, action = parents[state]
                    path.append(action)
                    state = previous
                return path[::-1]
            for action, nxt in self.neighbors(state):
                if nxt not in parents:
                    if len(parents) >= max_states:
                        raise RuntimeError("BFS state budget exhausted")
                    parents[nxt] = (state, action)
                    queue.append(nxt)
        return None


def make_environment(task):
    if task["benchmark"] == "sudoku":
        return SudokuEnv(task["puzzle"])
    if task["benchmark"] == "rush_hour":
        return RushHourEnv(task["board"])
    raise ValueError(f"unknown benchmark: {task['benchmark']}")

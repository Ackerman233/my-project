"""Local-contract adapters using public observations, never hidden solutions."""
import hashlib
import json
import re
from dynamic_skill.evidence.queue_revision import Proof, Truth
from dynamic_skill.environments.puzzles import RushHourEnv, SudokuEnv

ENTITY = r'[a-z][a-z0-9]* [0-9]+'


class Adapter:
    def __init__(self, public, *, observed_id=None):
        self.snapshot_id = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
        self.observed_id = self.snapshot_id if observed_id is None else observed_id
    def proof(self, value, reason, authoritative=True):
        return Proof(value, self.observed_id, 'public observation and declared local action contract', authoritative, reason)


class AlfworldEvidence(Adapter):
    """Only supported exact action grammars; negative lists require completeness."""
    def __init__(self, observation, admissible, *, complete=False, observed_id=None):
        super().__init__({'observation': observation, 'admissible': sorted(admissible), 'complete': complete}, observed_id=observed_id)
        self.observation, self.admissible, self.complete = observation, frozenset(admissible), complete
    def executable(self, action):
        opened = re.fullmatch(r'open (' + ENTITY + ')', action)
        surface = opened and re.search(r'On the ' + re.escape(opened[1]) + r', you see ', self.observation)
        if surface and action in self.admissible:
            return self.proof(Truth.UNKNOWN, 'conflicting public surface description and open permission', False)
        if action in self.admissible:
            return self.proof(Truth.TRUE, 'exact command publicly admitted')
        if surface:
            return self.proof(Truth.FALSE, 'latest feedback explicitly describes target as a surface')
        supported = re.fullmatch(r'(?:go to|open|examine) ' + ENTITY, action) or re.fullmatch(
            r'(?:take ' + ENTITY + r' from|move ' + ENTITY + r' to) ' + ENTITY, action)
        if self.complete and supported:
            return self.proof(Truth.FALSE, 'supported command absent from a declared complete current list')
        return self.proof(Truth.UNKNOWN, 'negative command membership is incomplete or grammar unsupported', False)
    def independent_after_skip(self, skipped, following):
        if re.fullmatch(r'go to ' + ENTITY, following):
            return self.proof(Truth.TRUE, 'absolute navigation is independent of omitted open/take/navigation',
                              bool(re.fullmatch(r'(?:go to|open) ' + ENTITY, skipped) or
                                   re.fullmatch(r'take ' + ENTITY + r' from ' + ENTITY, skipped)))
        opened = re.fullmatch(r'open (' + ENTITY + ')', skipped)
        take = re.fullmatch(r'take (' + ENTITY + ') from (' + ENTITY + ')', following)
        if opened and take and opened[1] == take[2]:
            return self.proof(Truth.TRUE if following in self.admissible else Truth.FALSE,
                              'take requires accessibility; effect must already be supported by current commands')
        pickup = re.fullmatch(r'take (' + ENTITY + ') from (' + ENTITY + ')', skipped)
        move = re.fullmatch(r'move (' + ENTITY + ') to (' + ENTITY + ')', following)
        if pickup and move and pickup[1] == move[1]:
            return self.proof(Truth.TRUE if following in self.admissible else Truth.FALSE,
                              'move requires possession; failed pickup cannot establish it')
        navigation = re.fullmatch(r'go to (' + ENTITY + ')', skipped)
        if navigation and opened is None and following == 'open ' + navigation[1]:
            return self.proof(Truth.TRUE if following in self.admissible else Truth.FALSE,
                              'open requires being at target; redundant navigation may already be satisfied')
        return self.proof(Truth.UNKNOWN, 'unsupported dependency contract', False)


def puzzle_rows(render):
    rows = [line.split(': ', 1)[1].split() for line in render.splitlines() if re.match(r'r[1-9]: ', line)]
    if not rows:
        raise ValueError('missing public board rows')
    return rows


def cell_command(action):
    match = re.fullmatch(r'(set)\(\s*([1-9])\s*,\s*([1-9])\s*,\s*([1-9])\s*\)', action)
    if not match:
        match = re.fullmatch(r'(clear)\(\s*([1-9])\s*,\s*([1-9])\s*\)', action)
    return match


class RushHourEvidence(Adapter):
    def __init__(self, render, *, observed_id=None):
        super().__init__({'render': render}, observed_id=observed_id)
        rows = puzzle_rows(render)
        if len(rows) != 6 or any(len(r) != 6 for r in rows):
            raise ValueError('expected complete public Rush Hour board')
        self.board = ''.join(''.join(r) for r in rows)
        RushHourEnv(self.board)  # Validate representation without searching for a solution.
    def executable(self, action):
        valid = RushHourEnv(self.board).step(action).valid
        return self.proof(Truth.TRUE if valid else Truth.FALSE, 'exact local slide contract on complete public board')
    def independent_after_skip(self, skipped, following):
        a, b = re.fullmatch(r'([A-Z])[+-][1-5]', skipped), re.fullmatch(r'([A-Z])[+-][1-5]', following)
        if not a or not b:
            return self.proof(Truth.UNKNOWN, 'unknown vehicle command dependency', False)
        if a[1] == b[1]:
            return self.proof(Truth.FALSE, 'relative move of same vehicle depends on omitted position update')
        return self.proof(Truth.TRUE, 'different vehicle; following path still requires fresh legality proof')


class SudokuEvidence(Adapter):
    def __init__(self, render, *, observed_id=None):
        super().__init__({'render': render}, observed_id=observed_id)
        self.rows = puzzle_rows(render)
        if len(self.rows) != 9 or any(len(r) != 9 for r in self.rows):
            raise ValueError('expected complete public Sudoku board')
        self.env()  # Check public clue/user-entry distinction.
    def env(self):
        clues = [[0 if v == '.' or v.endswith('*') else int(v) for v in row] for row in self.rows]
        probe = SudokuEnv(clues)
        for r, row in enumerate(self.rows):
            for c, value in enumerate(row):
                if value.endswith('*'):
                    result = probe.step(f'set({r+1},{c+1},{int(value[:-1])})')
                    if not result.valid:
                        raise ValueError('inconsistent public user entries')
        return probe
    def executable(self, action):
        valid = self.env().step(action).valid
        return self.proof(Truth.TRUE if valid else Truth.FALSE, 'local set/clear contract; not global solvability')
    def independent_after_skip(self, skipped, following):
        a, b = cell_command(skipped), cell_command(following)
        if not a or not b:
            return self.proof(Truth.UNKNOWN, 'unknown cell command dependency', False)
        if (a[2], a[3]) == (b[2], b[3]):
            if a[1] == 'clear' and b[1] == 'set' and self.rows[int(a[2])-1][int(a[3])-1] == '.':
                return self.proof(Truth.TRUE, 'empty-cell effect of redundant clear already holds')
            return self.proof(Truth.FALSE, 'same-cell sequence requires omitted write or clear effect')
        return self.proof(Truth.TRUE, 'different cell; shared unit conflicts checked by fresh local legality')

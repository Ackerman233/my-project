import unittest
from dynamic_skill.puzzles import SudokuEnv
from dynamic_skill.alfworld_rollout import parse_action_chunk
from experiments.puzzle_pilot import Settings, planner_system_prompt
from experiments.brief_protocol_pair import brief_prompt

class BriefTests(unittest.TestCase):
    def test_action_parser_ignores_reason(self):
        s='<reason>Fill the forced digit.</reason><actions>["set(1,2,7)"]</actions>'
        self.assertEqual(parse_action_chunk(s,1),['set(1,2,7)'])
    def test_only_terminal_output_section_replaced(self):
        env=SudokuEnv([[0]*9 for _ in range(9)])
        p=brief_prompt(env,[],6,1,'# Skill\nUNCHANGED KNOWLEDGE\n## Output\nOLD FORMAT')
        self.assertIn('# Skill\nUNCHANGED KNOWLEDGE',p)
        self.assertNotIn('OLD FORMAT',p)
        self.assertIn('## Remaining Steps\n6',p)
        self.assertIn('at most 30 words',p)
    def test_system_keeps_protocols_separate(self):
        a=planner_system_prompt(Settings(action_only=True),1)
        b=planner_system_prompt(Settings(action_only=False,output_protocol='brief_reason_v1'),1)
        self.assertIn('Output ONLY <actions>',a)
        self.assertIn('<reason>',b)
        self.assertNotIn('No explanation',b)

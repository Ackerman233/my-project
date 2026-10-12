import copy
import tempfile
import unittest
from pathlib import Path
from dynamic_skill.campaign_budget import CampaignBudget
from dynamic_skill.puzzles import SudokuEnv, RushHourEnv
from experiments.paper_cold_start_gate import authorize, paper_prompt


class PaperColdTests(unittest.TestCase):
    def test_authorization_preserves_history_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = CampaignBudget(Path(tmp) / 'budget.json', deadline=1)
            d = b.read()
            d['requests'] = [{'status': 'unknown_usage', 'budget_charge_usd': .5}]
            d['billing_checkpoint'] = {'user_reported_total_usd': .44, 'unsettled_reserve_usd': .5, 'request_count': 1}
            old = copy.deepcopy(d)
            authorize(d, 100)
            self.assertEqual(d['limit_usd'], 20)
            self.assertEqual(d['requests'], old['requests'])
            self.assertEqual(d['billing_checkpoint'], old['billing_checkpoint'])
            self.assertEqual(d['max_requests'], 37)
            self.assertEqual(d['max_unknown_requests'], 2)
            with self.assertRaises(RuntimeError): authorize(d, 101)

    def test_prompt_visible_state_only_and_remaining(self):
        env = SudokuEnv([[0] * 9 for _ in range(9)])
        p = paper_prompt(env, [{'action': 'set(1,1,1)', 'valid': True, 'feedback': 'accepted'}], 4, 1, 'BASIC SKILL')
        self.assertIn('## Remaining Steps\n4', p)
        self.assertIn(env.render(), p)
        self.assertIn('BASIC SKILL', p)
        self.assertNotIn('answer', p)
        self.assertNotIn('<think>', p)
        with self.assertRaises(ValueError): paper_prompt(env, [], 4, 2, '')

    def test_run_episode_uses_prompt_builder(self):
        from experiments.puzzle_pilot import Settings, run_episode
        task = {'id': 'test-paper', 'benchmark': 'rush_hour', 'board': '.'*12 + 'AA....' + '.'*18, 'minimum_actions': 1}
        seen = []
        def planner(p):
            seen.append(p)
            return {'response': '<actions>["A+4"]</actions>', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
        with tempfile.TemporaryDirectory() as tmp:
            r = run_episode(task, 'react', Settings(action_only=True, max_steps=6), Path(tmp), planner,
                            base_skill_text='FROZEN SKILL', prompt_builder=paper_prompt)
        self.assertTrue(r['success'])
        self.assertIn('## Remaining Steps\n6', seen[0])
        self.assertEqual(seen[0].count('Current Observation'), 1)
        self.assertNotIn('Use several actions', seen[0])


if __name__ == '__main__': unittest.main()

import unittest
from unittest.mock import patch
import json
from experiments.recovered_brief_protocol_pair import plan_jobs

class RecoveredTests(unittest.TestCase):
    def test_exact_remaining_and_retry(self):
        cfg={'run_order':[['done','brief_reason'],['brief-sudoku-2','brief_reason'],['brief-sudoku-3','brief_reason']]+[[str(i),'action_only'] for i in range(9)]}
        prev={'brief_reason':[{'task_id':'done','status':'success'},{'task_id':'brief-sudoku-2','status':'api_error'}]}
        later={'brief_reason':[{'task_id':'brief-sudoku-3','status':'api_error'}]}
        with patch('pathlib.Path.read_text',return_value=json.dumps(later)):
            jobs=plan_jobs(cfg,prev)
        self.assertEqual(len(jobs),10)
        self.assertEqual(jobs[-1],('brief-sudoku-2','brief_reason','infra_retry_once'))
        self.assertNotIn('brief-sudoku-3',[t for t,_,_ in jobs])

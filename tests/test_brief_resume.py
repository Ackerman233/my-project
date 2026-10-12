import unittest
from experiments.resume_brief_protocol_pair import plan_jobs

class ResumeTests(unittest.TestCase):
    def test_only_missing_then_one_infra_retry(self):
        cfg={'run_order':[['done','brief_reason'],['brief-sudoku-2','brief_reason'],['missing','action_only']]}
        previous={'brief_reason':[{'task_id':'done','status':'success'},
                                  {'task_id':'brief-sudoku-2','status':'api_error'}],'action_only':[]}
        self.assertEqual(plan_jobs(cfg,previous),[('missing','action_only','unattempted'),
            ('brief-sudoku-2','brief_reason','infra_retry_once')])
    def test_unexpected_error_not_retried(self):
        with self.assertRaises(RuntimeError):plan_jobs({'run_order':[]},{})

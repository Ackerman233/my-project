"""Offline integration tests: fake API outputs are never research scores."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from experiments import controller_research as r
from dynamic_skill.campaign_budget import CampaignBudget
from dynamic_skill.research_controller import POLICIES
from dataclasses import asdict


class FakeAPI:
    calls=0
    def __init__(self,*args,**kwargs): pass
    def __call__(self,prompt):
        FakeAPI.calls+=1
        return {'response':'<reason>Slide the target to the exit.</reason><actions>["A+4"]</actions>',
                'usage':{'prompt_tokens':100,'completion_tokens':30,'completion_tokens_details':{'reasoning_tokens':0}}}
    def close(self): pass


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run'; self.root.mkdir()
        self.ledger=Path(self.temp.name)/'budget.json'
        cfg={'splits':{'dev':[{'id':'unit-only','benchmark':'rush_hour','board':'............AA......................'}]},
             'skills':{'rush_hour':'Use legal slides.'},'system':r.SYSTEM,'settings':asdict(r.settings()),
             'policies':{k:asdict(v) for k,v in POLICIES.items()}}
        cfg['hash']=r.sha(cfg); r.save(self.root/'config.json',cfg)
        b=CampaignBudget(self.ledger,deadline=min(time.time()+600,r.DEADLINE),limit=30)
        d=b.read(); d[r.AUTH]={'config_sha':cfg['hash'],'initial_unknown':0}; b.write(d)
        for name,value in [('ROOT',self.root),('LEDGER',self.ledger),('ApiPlanner',FakeAPI)]:
            p=patch.object(r,name,value); p.start(); self.addCleanup(p.stop)
        FakeAPI.calls=0

    def test_real_environment_and_idempotence(self):
        args=['--split','dev','--policy','interrupt','--index','0']
        r.run(args); r.run(args)
        row=json.loads((self.root/'dev/interrupt/unit-only/result.json').read_text())
        self.assertTrue(row['success']); self.assertEqual(row['paid_requests'],1)
        self.assertEqual(FakeAPI.calls,1)
        self.assertEqual(json.loads(self.ledger.read_text())['requests'][0]['status'],'returned_usage')

    def test_unknown_usage_stops_following_runs(self):
        def unknown(*a): return {'response':'<actions>["A+4"]</actions>','usage':None}
        with patch.object(FakeAPI,'__call__',unknown):
            r.run(['--split','dev','--policy','interrupt','--index','0'])
        with self.assertRaisesRegex(RuntimeError,'unknown usage'):
            r.run(['--split','dev','--policy','react','--index','0'])

    def test_changed_config_rejected(self):
        cfg=json.loads((self.root/'config.json').read_text()); cfg['system']='changed'
        r.save(self.root/'config.json',cfg)
        with self.assertRaises(AssertionError): r.run(['--split','dev','--policy','react','--index','0'])

    def test_native_reasoning_stops(self):
        def thinking(*a): return {'response':'answer','usage':{'prompt_tokens':100,'completion_tokens':20,
                                          'completion_tokens_details':{'reasoning_tokens':10}}}
        with patch.object(FakeAPI,'__call__',thinking):
            r.run(['--split','dev','--policy','interrupt','--index','0'])
        row=json.loads((self.root/'dev/interrupt/unit-only/result.json').read_text())
        self.assertEqual(row['status'],'thinking_policy_stop'); self.assertEqual(row['environment_steps'],0)

if __name__=='__main__': unittest.main()

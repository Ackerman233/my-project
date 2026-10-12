import json
from experiments import alfworld_staged_recovery as s

def test_config_split_selection_frozen():
    assert len(s.TYPES)==len(set(s.TYPES))==6

def test_independent_gate_rejects_tie(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    x={'complete':True,'successes':1,'rows':[{'success':True,'model_calls':3,'recovery_calls':0}]}
    y={'complete':True,'successes':1,'rows':[{'success':True,'model_calls':2,'recovery_calls':1}]}
    monkeypatch.setattr(s,'summary',lambda cfg:{'validation':{'dynamic_interrupt':x,'recovery':y}})
    assert not s.gate({'hash':'frozen'})

def test_gate_rejects_regression(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    x={'complete':True,'successes':1,'rows':[{'success':True,'model_calls':3},{'success':False,'model_calls':4},{'success':False,'model_calls':4}]}
    y={'complete':True,'successes':2,'rows':[{'success':False,'model_calls':3,'recovery_calls':1},{'success':True,'model_calls':4,'recovery_calls':1},{'success':True,'model_calls':4,'recovery_calls':1}]}
    monkeypatch.setattr(s,'summary',lambda cfg:{'validation':{'dynamic_interrupt':x,'recovery':y}})
    assert not s.gate({'hash':'frozen'})

def test_gate_accepts_extra_success(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path)
    x={'complete':True,'successes':1,'rows':[{'success':True,'model_calls':3},{'success':False,'model_calls':4}]}
    y={'complete':True,'successes':2,'rows':[{'success':True,'model_calls':3,'recovery_calls':1},{'success':True,'model_calls':4,'recovery_calls':1}]}
    monkeypatch.setattr(s,'summary',lambda cfg:{'validation':{'dynamic_interrupt':x,'recovery':y}})
    assert s.gate({'hash':'frozen'})

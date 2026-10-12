import json,time
from pathlib import Path
import pytest
from experiments import interrupt_recovery_ablation as a

def test_one_factor_arms():
    assert {k for k in a.ARMS['A'] if a.ARMS['A'][k]!=a.ARMS['B'][k]}=={'dynamic'}
    assert {k for k in a.ARMS['B'] if a.ARMS['B'][k]!=a.ARMS['C'][k]}=={'history_evidence'}
    assert {k for k in a.ARMS['C'] if a.ARMS['C'][k]!=a.ARMS['D'][k]}=={'state_check'}

def test_prepared_config_cannot_authorize_paid_work(tmp_path,monkeypatch):
    from dynamic_skill.campaign_budget import CampaignBudget
    root=tmp_path/'trial';root.mkdir();campaign=tmp_path/'budget';campaign.mkdir()
    (root/'config.json').write_text('{}')
    budget=CampaignBudget(campaign/'budget.json',deadline=time.time()+60)
    before=(campaign/'budget.json').read_bytes()
    monkeypatch.setattr(a,'ROOT',root);monkeypatch.setattr(a,'CAMPAIGN',campaign)
    with pytest.raises(RuntimeError,match='NOT authorized'):a.advance()
    assert (campaign/'budget.json').read_bytes()==before

def test_empty_summary_is_not_zero_accuracy(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path);a.summarize()
    r=json.loads((tmp_path/'summary.json').read_text())
    assert not r['paid_experiments_run']
    assert all(x['success_rate'] is None and x['false_interrupt_count'] is None for x in r['arms'].values())

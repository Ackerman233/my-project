import json
import time
from pathlib import Path
import pytest

from dynamic_skill.infrastructure.campaign_budget import CampaignBudget, CampaignHalt
from experiments.common import bounded_campaign as batch


def setup_stage(tmp_path):
    path = tmp_path / 'budget.json'
    initial = CampaignBudget(path, deadline=time.time()+600, limit=batch.AUTHORIZED_LIMIT_USD)
    state = initial.read()
    state[batch.auth_key(tmp_path)] = {'cumulative_limit_usd':batch.AUTHORIZED_LIMIT_USD, 'closed':False,
        'deadline_unix':time.time()+600, 'max_requests':4, 'batch_spend_ceiling':1,
        'start_request_count':0, 'max_new_unknown':1}
    initial.write(state)
    return batch.StageBudget(tmp_path, path=path)


def test_new_batch_enforces_batch_ceiling_and_retains_unknown_reservation(tmp_path):
    budget = setup_stage(tmp_path)
    index = budget.reserve('actor', .6)
    budget.settle(index, response={'usage':{'prompt_tokens':1,'completion_tokens':1}})
    with pytest.raises(CampaignHalt, match='budget_stop'):
        budget.reserve('oversized', 1.1)
    index = budget.reserve('optimizer', .7)
    budget.settle(index, error=TimeoutError())
    state = budget.read()
    assert state['requests'][-1]['budget_charge_usd'] == .7
    with pytest.raises(CampaignHalt, match='unknown_cost_stop'):
        budget.reserve('no retry', .1)
    assert len(budget.read()['requests']) == 2


@pytest.mark.parametrize('previous_limit', [95, 105, 160])
def test_limit_raise_keeps_spend_and_unknowns_and_does_not_open_requests(tmp_path, monkeypatch, previous_limit):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    monkeypatch.setattr(batch, 'CAMPAIGN', campaign)
    ledger = CampaignBudget(campaign/'budget.json',deadline=time.time()+600,limit=previous_limit)
    index = ledger.reserve('old', .5)
    ledger.settle(index,error=TimeoutError())
    before = ledger.read()
    batch.raise_authorized_limit(tmp_path/'audit')
    after = ledger.read()
    assert after['limit_usd'] == batch.AUTHORIZED_LIMIT_USD and after['requests'] == before['requests']
    assert ledger.effective_spend(after) == ledger.effective_spend(before)
    assert after['max_requests'] == len(after['requests'])
    assert after['deadline_unix'] <= time.time()
    assert after['limit_change_history'][-1]['user_authorization'] == batch.USER_AUTHORIZATION


@pytest.mark.parametrize('change', ['closed', 'limit_changed'])
def test_closed_or_changed_authorization_cannot_spend(tmp_path, change):
    budget = setup_stage(tmp_path)
    state = budget.read()
    if change == 'closed':state[batch.auth_key(tmp_path)]['closed'] = True
    else:state['limit_usd'] = 106
    budget.write(state)
    with pytest.raises(CampaignHalt):budget.reserve('actor')
    assert not budget.read()['requests']


def test_missing_ledger_cannot_reset_budget(tmp_path):
    with pytest.raises(ValueError, match='Existing cumulative ledger'):
        batch.StageBudget(tmp_path, path=tmp_path/'missing.json')


def test_error_rearm_requires_expired_cooldown_and_keeps_unknown_charges(tmp_path,monkeypatch):
    campaign=tmp_path/'campaign';campaign.mkdir();monkeypatch.setattr(batch,'CAMPAIGN',campaign)
    budget=CampaignBudget(campaign/'budget.json',deadline=time.time()+100,limit=160)
    state=budget.read();state.update(consecutive_errors=2,cooldown_until=time.time()+60);budget.write(state)
    health=tmp_path/'health.json';health.write_text(json.dumps({'paid_requests':0,'route':'direct',
        'finished_unix':time.time(),'probes':[{'status':404}]*3}))
    with pytest.raises(ValueError,match='Cooldown'):batch.rearm_after_cooldown(tmp_path/'audit',health)
    state['cooldown_until']=time.time()-1;budget.write(state)
    before=budget.read();batch.rearm_after_cooldown(tmp_path/'audit',health);after=budget.read()
    assert after['requests']==before['requests'] and after['limit_usd']==before['limit_usd']
    assert after['consecutive_errors']==0 and after['max_requests']==len(after['requests'])
    assert after['error_guard_recovery_history'][-1]['unknown_reservations_retained']

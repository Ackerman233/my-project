import pytest
from dynamic_skill.campaign_budget import CampaignBudget


def test_billing_checkpoint_preserves_unknown_reserve_and_new_cost():
    data={'requests':[{'budget_charge_usd':4.0},{'budget_charge_usd':.2}],
          'billing_checkpoint':{'user_reported_total_usd':.44,'unsettled_reserve_usd':1.5,'request_count':1}}
    assert CampaignBudget.effective_spend(data)==pytest.approx(2.14)
    assert data['requests'][0]['budget_charge_usd']==4.0


def test_bounded_reservation_retained_when_usage_unknown(tmp_path):
    from dynamic_skill.campaign_budget import CampaignHalt
    budget=CampaignBudget(tmp_path/'budget.json',deadline=100,clock=lambda:1,limit=.3)
    with pytest.raises(CampaignHalt,match='budget_stop'):
        budget.reserve('default-too-large')
    request=budget.reserve('bounded-request',reservation_usd=.2)
    budget.settle(request,error=ConnectionError())
    assert budget.read()['requests'][0]['budget_charge_usd']==.2
    with pytest.raises(CampaignHalt,match='budget_stop'):
        budget.reserve('no-room',reservation_usd=.2)

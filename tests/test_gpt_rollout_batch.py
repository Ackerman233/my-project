import pytest
from dynamic_skill.campaign_budget import CampaignBudget
from experiments.gpt_rollout_batch import authorize


def test_authorization_preserves_budget_and_errors(tmp_path):
    budget = CampaignBudget(tmp_path/'budget.json', deadline=100, clock=lambda:1)
    for i in range(7):
        budget.settle(budget.reserve('prior'), {'usage':{'prompt_tokens':1,'completion_tokens':1}})
    old = budget.read()
    authorize(budget)
    new = budget.read()
    assert new['requests'] == old['requests']
    for k in ('limit_usd','max_unknown_requests','cooldown_until','policy_violation','reservation_usd'):
        assert new[k] == old[k]
    assert new['max_requests'] == 295
    assert new['deadline_unix'] == new['rollout_authorization']['at_unix'] + 3600
    with pytest.raises(RuntimeError):
        authorize(budget)

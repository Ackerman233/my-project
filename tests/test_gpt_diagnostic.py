import pytest
from dynamic_skill.campaign_budget import CampaignBudget, CampaignHalt
from dynamic_skill.puzzles import make_environment
from experiments.gpt_recovery_diagnostic import request_options, evaluate


def test_gpt_wire_contract():
    options = request_options()
    assert options['model'] == 'gpt-5.4' and options['reasoning_effort'] == 'none'
    assert options['max_completion_tokens'] == 512
    assert 'enable_thinking' not in options and 'max_tokens' not in options


def test_custom_budget_persists_rates_and_unknown_stop(tmp_path):
    budget = CampaignBudget(tmp_path / 'b.json', deadline=100, clock=lambda: 1,
                            input_rate=25, output_rate=150, max_requests=6, max_unknown=1)
    budget.settle(budget.reserve('ok'), {'usage': {'prompt_tokens': 1000, 'completion_tokens': 100}})
    assert budget.read()['requests'][0]['budget_charge_usd'] == pytest.approx(.04)
    resumed = CampaignBudget(budget.path, deadline=200, input_rate=1)
    assert resumed.read()['input_usd_per_million'] == 25
    budget.settle(budget.reserve('error'), error=TimeoutError())
    with pytest.raises(CampaignHalt, match='unknown_cost_stop'):
        budget.reserve('no more')


def test_probe_replay_is_isolated():
    env = make_environment({'benchmark': 'rush_hour', 'board': '......' '......' 'AA....' '......' '......' '......'})
    result = evaluate(env, '<actions>["A+4"]</actions>', 1)
    assert result['chunk_reaches_goal'] and result['first_action_valid'] and not env.solved


def test_remaining_authorization_only_changes_cap_and_audit(tmp_path):
    from experiments.gpt_recovery_diagnostic import authorize_remaining_probes
    budget = CampaignBudget(tmp_path / 'b.json', deadline=100, clock=lambda: 1, max_requests=3)
    for _ in range(3):
        budget.settle(budget.reserve('prior'), {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    previous = budget.read()
    authorize_remaining_probes(budget)
    updated = budget.read()
    assert updated.pop('max_requests') == 7
    updated.pop('continuation_authorizations')
    previous.pop('max_requests')
    assert updated == previous
    with pytest.raises(RuntimeError):
        authorize_remaining_probes(budget)


def test_empty_action_list_is_output_failure():
    env = make_environment({'benchmark': 'rush_hour', 'board': '......' '......' 'AA....' '......' '......' '......'})
    with pytest.raises(ValueError):
        evaluate(env, '<actions>[]</actions>', 4)


def test_retry_authorization_preserves_cost_and_is_single_use(tmp_path):
    from experiments.gpt_recovery_diagnostic import authorize_single_retry
    budget = CampaignBudget(tmp_path / 'b.json', deadline=100, clock=lambda: 1, max_unknown=1)
    budget.settle(budget.reserve('first'), {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    APIConnectionError = type('APIConnectionError', (Exception,), {})
    budget.settle(budget.reserve('failed'), error=APIConnectionError())
    previous = budget.read()
    authorize_single_retry(budget)
    assert budget.read()['requests'] == previous['requests']
    assert budget.read()['deadline_unix'] == previous['deadline_unix']
    assert budget.read()['limit_usd'] == 5
    with pytest.raises(RuntimeError):
        authorize_single_retry(budget)
    budget.settle(budget.reserve('retry'), {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    with pytest.raises(CampaignHalt, match='request_cap_stop'):
        budget.reserve('unauthorized extra request')

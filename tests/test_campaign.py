import json
from pathlib import Path
import pytest
from dynamic_skill.campaign_budget import CampaignBudget, CampaignHalt, BudgetPlanner
from experiments.puzzle_pilot import Settings, run_episode
from experiments.prepare_curriculum import prepare
from dynamic_skill.puzzles import make_environment


def test_reserve_settle_and_missing_usage_is_not_free(tmp_path):
    budget = CampaignBudget(tmp_path / 'budget.json', deadline=100, clock=lambda: 10)
    i = budget.reserve('ok')
    assert budget.read()['requests'][i]['budget_charge_usd'] == .5
    budget.settle(i, {'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})
    assert budget.read()['requests'][i]['budget_charge_usd'] == pytest.approx(.00015)
    j = budget.reserve('timeout')
    budget.settle(j, error=TimeoutError())
    assert budget.read()['requests'][j]['budget_charge_usd'] == .5
    assert budget.read()['requests'][j]['status'] == 'unknown_usage'
    with pytest.raises(AssertionError):
        budget.settle(j, {})


def test_no_reset_on_resume_and_pending_reservation_limits(tmp_path):
    path = tmp_path / 'budget.json'
    budget = CampaignBudget(path, deadline=100, limit=.5, clock=lambda: 10)
    budget.reserve('pending')
    resumed = CampaignBudget(path, deadline=9999, limit=999, clock=lambda: 10)
    assert resumed.read()['limit_usd'] == .5 and resumed.read()['deadline_unix'] == 100
    with pytest.raises(CampaignHalt, match='budget_stop'):
        resumed.reserve('cannot overspend')


def test_deadline_and_cooldown_and_policy(tmp_path):
    now = [10]
    budget = CampaignBudget(tmp_path / 'budget.json', deadline=1000, clock=lambda: now[0])
    for _ in range(2):
        budget.settle(budget.reserve('fail'), error=TimeoutError())
    with pytest.raises(CampaignHalt, match='api_cooldown'):
        budget.reserve('wait')
    now[0] = 920
    budget.settle(budget.reserve('bad policy'), {'reasoning_content': 'unexpected',
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    with pytest.raises(CampaignHalt, match='thinking_policy_stop'):
        budget.reserve('no more paid calls')
    now[0] = 1001
    with pytest.raises(CampaignHalt, match='deadline_stop'):
        budget.reserve('too late')


def test_four_unknown_calls_stop_after_cooldowns(tmp_path):
    now = [1]
    budget = CampaignBudget(tmp_path / 'budget.json', deadline=99999, clock=lambda: now[0])
    for _ in range(4):
        budget.settle(budget.reserve('fail'), error=TimeoutError())
        now[0] += 1000
    with pytest.raises(CampaignHalt, match='unknown_cost_stop'):
        budget.reserve('blocked')


def test_runner_guard_has_no_phantom_calls(tmp_path):
    budget = CampaignBudget(tmp_path / 'budget.json', deadline=1, clock=lambda: 2)
    def forbidden(prompt):
        raise AssertionError('must not call API after deadline')
    planner = BudgetPlanner(forbidden, budget, 'test')
    task = {'id': 'test', 'benchmark': 'rush_hour', 'minimum_actions': 1,
            'board': '......' '......' 'AA....' '......' '......' '......'}
    row = run_episode(task, 'react', Settings(), tmp_path / 'run', planner)
    assert row['status'] == 'deadline_stop' and row['model_calls'] == row['environment_steps'] == 0
    assert not budget.read()['requests']


def test_curriculum_solutions_are_offline_and_exact(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'data/puzzle_pilot_20260927'
    out = tmp_path / 'curriculum'
    prepare(source, out)
    tasks = json.loads((out / 'tasks.json').read_text())
    private = {r['id']: r for r in json.loads((out / 'validation_private.json').read_text())}
    assert len(tasks) == 16
    assert set(t['split'] for t in tasks) == {'dev', 'validation'}
    for task in tasks:
        assert 'solution' not in task and 'oracle_actions' not in task
        env = make_environment(task)
        if task['benchmark'] == 'rush_hour':
            actions = private[task['id']]['oracle_actions']
        else:
            sol = private[task['id']]['solution']
            actions = [f'set({r+1},{c+1},{sol[r][c]})' for r in range(9) for c in range(9) if task['puzzle'][r][c] == 0]
        assert len(actions) == task['minimum_actions']
        for action in actions:
            assert env.step(action).valid
        assert env.solved

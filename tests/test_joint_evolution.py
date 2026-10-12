from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import time
import subprocess
import sys
import pytest

from experiments.alfworld.joint_evolution.schema import Genome, InterruptPolicy, apply_proposal, validate_splits
from experiments.alfworld.joint_evolution.engine import evolve, paired_gate
from experiments.alfworld.joint_evolution.demo import splits, scripted_evaluator, scripted_proposer, run_demo
from experiments.alfworld.joint_evolution.runtime import LiveTransport
from experiments.common.puzzle_pilot import Settings
from dynamic_skill.infrastructure.campaign_budget import CampaignBudget, CampaignHalt


def proposal(parent):
    return {'parent_hash': parent.hash, 'edits': [{'op': 'append', 'target': '', 'content': 'Preserve independent navigation.'}],
            'policy': {**asdict(parent.policy), 'invalid_patience': 2}, 'evidence_ids': ['train-only']}


def test_joint_candidate_uses_upstream_patch_and_changes_real_policy(monkeypatch):
    import skillopt.optimizer.skill as upstream
    original = upstream.apply_patch
    calls = []
    def spy(text, patch):
        calls.append(patch)
        return original(text, patch)
    monkeypatch.setattr(upstream, 'apply_patch', spy)
    parent = Genome('Original skill.')
    candidate = apply_proposal(proposal(parent), parent, [{'id': 'train-only'}])
    assert calls and candidate.skill_text != parent.skill_text
    assert candidate.policy.executable().invalid_patience == 2
    assert candidate.parent_hash == parent.hash and candidate.revision == 1


@pytest.mark.parametrize('mutation', ['stale_parent', 'heldout_evidence', 'unbounded_policy', 'unchanged_policy', 'unknown_code_field', 'missing_edit_target'])
def test_joint_candidate_rejects_invalid_or_leaked_proposals(mutation):
    parent = Genome('Original skill.'); raw = proposal(parent)
    if mutation == 'stale_parent':raw['parent_hash'] = 'old'
    if mutation == 'heldout_evidence':raw['evidence_ids'] = ['test-private']
    if mutation == 'unbounded_policy':raw['policy']['max_actions'] = 99
    if mutation == 'unchanged_policy':raw['policy'] = asdict(parent.policy)
    if mutation == 'unknown_code_field':raw['code'] = 'pass'
    if mutation == 'missing_edit_target':raw['edits'] = [{'op': 'replace', 'target': 'absent', 'content': 'new'}]
    with pytest.raises(ValueError):apply_proposal(raw, parent, [{'id': 'train-only'}])


def test_physical_split_leakage_is_rejected_even_with_renamed_task_ids():
    data = splits();data['confirmation'][1][0]['scene_id'] = data['test'][0]['scene_id']
    with pytest.raises(ValueError, match='physical scene'):validate_splits(data, 2)


def test_real_controller_two_round_loop_accepts_then_rejects_and_freezes_test(tmp_path):
    best, final = run_demo(tmp_path / 'run')
    history = json.loads((tmp_path / 'run/history.json').read_text())
    assert [r['accepted'] for r in history] == [True, False]
    assert best.revision == 1 and best.policy.invalid_patience == 2
    assert set(final) == {'initial', 'skill_only', 'policy_only', 'evolved'}
    assert final['initial'][0]['model_calls'] == final['skill_only'][0]['model_calls'] == 2
    assert final['policy_only'][0]['model_calls'] == final['evolved'][0]['model_calls'] == 1
    assert final['initial'][0]['actual_tail_interruptions'] == 1
    assert final['evolved'][0]['actual_tail_interruptions'] == 0
    assert json.loads((tmp_path / 'run/best/frozen_before_test.json').read_text())['genome_hash'] == best.hash
    for path in (tmp_path / 'run/generations').glob('*/candidates/*/prompt.json'):
        prompt = json.loads(path.read_text())['prompt']
        assert 'test-private' not in prompt and 'select-private' not in prompt and 'confirm-private' not in prompt


def test_confirmation_regression_keeps_incumbent_and_still_runs_test(tmp_path):
    parent = Genome('Original skill.')
    def evaluator(tasks, genome, directory):
        rows = scripted_evaluator(tasks, genome, directory)
        if 'confirmation' in directory.parts and genome.revision > 0:
            rows[0].update(success=False, status='step_limit')
        return rows
    best, _ = evolve(splits(1), parent, evaluator, scripted_proposer, tmp_path / 'run',
                     generations=1, candidates_per_generation=1)
    assert best == parent
    history = json.loads((tmp_path / 'run/history.json').read_text())
    assert history[0]['confirmation_gate']['losses'] == 1
    assert not history[0]['accepted']


def test_proposal_transport_error_stops_without_test_or_implicit_retry(tmp_path):
    calls = []
    def failed_proposer(prompt, directory):
        calls.append(prompt);raise CampaignHalt('api_error')
    with pytest.raises(CampaignHalt):
        evolve(splits(1), Genome('Skill.'), scripted_evaluator, failed_proposer,
               tmp_path / 'run', generations=1, candidates_per_generation=1)
    assert len(calls) == 1 and not (tmp_path / 'run/test').exists()
    assert json.loads((tmp_path / 'run/status.json').read_text())['status'] == 'stopped'


def test_gate_does_not_reward_cheap_failed_episodes_or_protocol_errors():
    def row(success, calls):
        return dict(task_id='x', success=success, status='success' if success else 'step_limit',
                    model_calls=calls, cost_usd=calls*.1, environment_steps=3, invalid_actions=0)
    assert not paired_gate([row(False, 8)], [row(False, 1)])['accept']
    assert not paired_gate([row(True, 8)], [{**row(True, 1), 'status':'format_error'}])['accept']
    assert paired_gate([row(True, 8)], [row(True, 2)])['accept']


def test_live_transport_charges_actor_and_optimizer_and_preserves_unknown(tmp_path, monkeypatch):
    import experiments.alfworld.joint_evolution.runtime as runtime
    budget = CampaignBudget(tmp_path / 'budget.json', deadline=time.time()+600, limit=5)
    class Api:
        def __init__(self,*args,**kwargs):pass
        def close(self):pass
        def __call__(self,prompt):
            return {'response':'{}','usage':{'prompt_tokens':2,'completion_tokens':1}}
    monkeypatch.setattr(runtime, 'ApiPlanner', Api)
    transport = LiveTransport(tmp_path, budget, Settings(max_tokens=128))
    transport.actor('actor')('public observation')
    transport.proposer(128)('train-only', tmp_path / 'candidate')
    data = budget.read()
    assert len(data['requests']) == 2
    assert {r['label'].split('/')[-1] for r in data['requests']} == {'actor','optimizer'}
    class Unknown(Api):
        def __call__(self,prompt):return {'response':'{}','usage':None}
    monkeypatch.setattr(runtime,'ApiPlanner',Unknown)
    with pytest.raises(CampaignHalt, match='unknown_cost_stop'):
        transport.request('unknown', system='system', max_tokens=128, label='unknown')
    row = budget.read()['requests'][-1]
    assert row['status'] == 'unknown_usage' and row['budget_charge_usd'] >= .5


def test_closed_budget_blocks_api_construction(tmp_path, monkeypatch):
    import experiments.alfworld.joint_evolution.runtime as runtime
    budget = CampaignBudget(tmp_path / 'budget.json', deadline=0)
    def forbidden(*args,**kwargs):raise AssertionError('API client must not be created')
    monkeypatch.setattr(runtime,'ApiPlanner',forbidden)
    with pytest.raises(CampaignHalt, match='deadline_stop'):
        LiveTransport(tmp_path,budget,Settings()).request('x',system='x',max_tokens=128,label='x')
    assert not budget.read()['requests']


def test_legacy_imports_share_canonical_module_and_mutable_globals(monkeypatch):
    old = importlib.import_module('experiments.alfworld_campaign')
    new = importlib.import_module('experiments.alfworld.common.alfworld_campaign')
    assert old is new
    monkeypatch.setattr(old,'SYSTEM','marker')
    assert new.SYSTEM == 'marker'
    old = importlib.import_module('dynamic_skill.skill_executor')
    new = importlib.import_module('dynamic_skill.execution.skill_executor')
    assert old is new and old.ChunkExecutor is new.ChunkExecutor


def live_config_fixture(tmp_path):
    data = splits(1)
    for group in [data['train'], data['selection'], *data['confirmation'], data['test']]:
        for task in group:
            path = tmp_path / 'valid_unseen' / task['scene_id'] / 'trial1' / 'game.tw-pddl'
            path.parent.mkdir(parents=True)
            path.write_text('fixture content ' + task['id'])
            task['gamefile'] = str(path)
    return {'splits': data, 'generations': 1, 'candidates_per_generation': 1,
            'settings': asdict(Settings(max_tokens=128)), 'initial': asdict(Genome('Original skill.'))}


def test_live_validation_freezes_physical_identity_and_blocks_scene_exposure(tmp_path, monkeypatch):
    from experiments.alfworld.joint_evolution.cli import validate_config
    monkeypatch.chdir(tmp_path)
    cfg = live_config_fixture(tmp_path)
    frozen = validate_config(cfg)
    assert frozen['splits']['test'][0]['game_sha256']
    assert frozen['exposure_audit']['independent_evaluation_claim_allowed']
    (tmp_path / 'results/old').mkdir(parents=True)
    # Same physical scene under a different root/trial is already exposed.
    (tmp_path / 'results/old/result.json').write_text(json.dumps({'gamefile': '/elsewhere/test-private/trial2/game.tw-pddl'}))
    with pytest.raises(ValueError, match='prior exposure'):
        validate_config(cfg)
    cfg['allow_previously_exposed_evaluation'] = True
    assert not validate_config(cfg)['exposure_audit']['independent_evaluation_claim_allowed']
    task = cfg['splits']['test'][0]
    Path(task['gamefile']).write_text('changed after freeze')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_config(frozen, check_exposure=False)


def test_live_validation_rejects_content_duplicates_under_new_scene_names(tmp_path, monkeypatch):
    from experiments.alfworld.joint_evolution.cli import validate_config
    monkeypatch.chdir(tmp_path)
    cfg = live_config_fixture(tmp_path)
    Path(cfg['splits']['test'][0]['gamefile']).write_bytes(Path(cfg['splits']['train'][0]['gamefile']).read_bytes())
    with pytest.raises(ValueError, match='Duplicate physical game content'):
        validate_config(cfg)


def test_live_run_closed_stage_does_not_open_batch_or_create_output(tmp_path, monkeypatch):
    from experiments.alfworld.joint_evolution import cli
    root = tmp_path / 'batch'
    monkeypatch.setattr(cli, 'CAMPAIGN', tmp_path)
    monkeypatch.setattr(cli, 'preflight', lambda _: {'authorization_open': False})
    def forbidden(*args):raise AssertionError('Closed stage must not load API credentials')
    monkeypatch.setattr(cli, 'open_batch', forbidden)
    with pytest.raises(ValueError, match='authorization is closed'):
        cli.run(root)
    assert not root.exists()


def test_legacy_module_cli_has_a_valid_argv_path():
    result = subprocess.run([sys.executable, '-m', 'experiments.react', '--help'],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_prepare_preflight_are_read_only_for_budget_and_detect_source_drift(tmp_path, monkeypatch):
    from experiments.alfworld.joint_evolution import cli
    from experiments.common.bounded_campaign import StageBudget
    monkeypatch.chdir(tmp_path)
    cfg = live_config_fixture(tmp_path)
    campaign = tmp_path / 'shared'
    campaign.mkdir()
    ledger = campaign / 'budget.json'
    CampaignBudget(ledger, deadline=0, limit=cli.AUTHORIZED_LIMIT_USD)
    for name in ('dynamic_skill/migration_20261008.json', 'experiments/migration_20261008.json', 'skillopt/skillopt/optimizer/skill.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    source = tmp_path / 'experiments/fixture.py'
    source.write_text('VALUE = 1\n')
    monkeypatch.setattr(cli, 'CAMPAIGN', campaign)
    monkeypatch.setattr(cli, 'StageBudget', lambda root: StageBudget(root, path=ledger))
    config_path = tmp_path / 'input.json'
    config_path.write_text(json.dumps(cfg))
    before = ledger.read_bytes()
    root = tmp_path / 'prepared'
    cli.prepare(config_path, root)
    result = cli.preflight(root)
    assert result['source_and_data_valid'] and not result['authorization_open']
    assert result['paid_requests'] == 0 and ledger.read_bytes() == before
    source.write_text('VALUE = 2\n')
    with pytest.raises(AssertionError):
        cli.preflight(root)

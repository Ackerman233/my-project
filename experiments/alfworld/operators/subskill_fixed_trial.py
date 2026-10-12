"""Fixed execution of the same four v5 development tasks; interrupt is the changed control."""
import argparse
import fcntl
import os
import shutil
import time
from pathlib import Path
from experiments import protocol_diagnostic_trial as base
from experiments.interrupt_recovery_ablation import read, digest, config_hash
from experiments.puzzle_pilot import ApiPlanner, Settings, save
from dynamic_skill.campaign_budget import BudgetPlanner
from dynamic_skill.control_only_evolution import ControlPolicy, run_control_episode
from dynamic_skill.space_subskills import SYSTEM_SKILL


def configure():
    base.ROOT = Path('results/subskill_fixed_20261007')
    base.AUTH = 'subskill_fixed_20261007_authorization'
    base.configure()
    base.trial.ARMS = {'F_fixed': {'macro_mode': True, 'discovery_interrupt': False, 'dynamic': False}}


def prepare():
    configure()
    source = Path('results/protocol_diagnostic_v5_20261007')
    assert read(source / 'diagnostic_audit.json')['protocol_gate_passed']
    cfg = read(source / 'config.json')
    # Algorithm, prompt, model, task text and costs must match the completed dynamic run.
    for path, sha in cfg['source_hashes'].items():
        if path != 'experiments/campaign_runner.py':
            assert digest(Path(path)) == sha, 'Changed frozen core: ' + path
    cfg.update(protocol='subskills-v5-fixed-paired-development', arms=base.trial.ARMS,
               job_order=[[i, 'F_fixed'] for i in range(len(cfg['tasks']))])
    cfg['common'].pop('dynamic')
    cfg['paired_dynamic_root'] = str(source)
    sources = sorted(set(cfg['source_hashes']) | {'experiments/subskill_fixed_trial.py'})
    cfg['source_hashes'] = {p: digest(Path(p)) for p in sources}
    cfg['note'] = ('Same full four-task v5 development set, no outcome-based task selection. '
        'Same prompt/skill/model/primitive limit/settings; fixed controller consumes nonterminal tails. '
        'Dynamic episodes were run earlier, so ordering and model variation limit causal inference. '
        'Not independent validation or original SkillOpt/ReAct comparison.')
    if (base.ROOT / 'config.json').exists():
        assert read(base.ROOT / 'config.json') == cfg
    else:
        save(base.ROOT / 'config.json', cfg)
        for path in sources:
            dst = base.ROOT / 'source_snapshot' / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    print('PREPARED', config_hash(cfg))


def advance():
    configure()
    if not os.environ.get('TARGET_AZURE_OPENAI_API_KEY'):
        os.environ['TARGET_AZURE_OPENAI_API_KEY'] = Path('/opt/data/private/api-key').read_text().strip()
    os.environ.setdefault('ALFWORLD_DATA', '/opt/data/private/data/alfworld')
    cfg = read(base.ROOT / 'config.json')
    with (base.trial.CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = base.trial.TrialBudget(base.trial.CAMPAIGN / 'budget.json', deadline=0)
        data = budget.read()
        auth = data[base.AUTH]
        assert config_hash(cfg) == auth['config_sha256'] and data['limit_usd'] == 85
        assert time.time() < auth['deadline_unix'] and not data['policy_violation']
        assert time.time() >= data['cooldown_until']
        assert not any(r['status'] == 'pending' for r in data['requests'])
        assert not any(r['status'] != 'returned_usage' or r.get('error_type')
                       for r in data['requests'][auth['start_request_count']:])
        assert all(digest(Path(p)) == h for p, h in cfg['source_hashes'].items())
        for index, arm in cfg['job_order']:
            task = cfg['tasks'][index]
            out = base.ROOT / 'online' / arm
            episode = out / 'predictions' / (cfg['model'] + '__' + task['id'] + '__fixed_chunk')
            if episode.exists():
                assert (episode / 'result.json').exists(), 'Incomplete episode requires audit'
                assert read(episode / 'result.json')['status'] in (
                    'success', 'model_call_limit', 'step_limit', 'wall_time_limit',
                    'environment_failure', 'interrupt_budget_exhausted', 'format_error')
                continue
            assert digest(Path(task['gamefile'])) == task['game_sha256']
            from skillopt.envs.alfworld.rollout import build_alfworld_env
            settings = Settings(**cfg['settings'])
            api = ApiPlanner(settings, system_prompt=SYSTEM_SKILL)
            env = build_alfworld_env(env_num=1, eval_dataset='eval_in_distribution', seed=42,
                                    is_train=False, specific_gamefiles=[task['gamefile']])
            before = budget.effective_spend(data)
            try:
                result = run_control_episode(env, task, settings, out,
                    BudgetPlanner(api, budget, str(episode)), cfg['task_skill'],
                    ControlPolicy(**cfg['policy']), **cfg['common'], **cfg['arms'][arm])
            finally:
                api.close()
                env.close()
            result.update(arm=arm, cost_usd=budget.effective_spend(budget.read()) - before)
            save(episode / 'result.json', result)
            base.trial.summary()
            return
        base.trial.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'authorize', 'summary'])
    args = parser.parse_args()
    configure()
    {'prepare': prepare, 'authorize': base.authorize, 'summary': base.trial.summary}[args.action]()

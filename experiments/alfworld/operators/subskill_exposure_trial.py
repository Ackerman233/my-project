"""Six new development tasks to measure real multi-action and interrupt exposure."""
import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path
from experiments import protocol_diagnostic_trial as base
from experiments.interrupt_recovery_ablation import read, digest, config_hash
from experiments.puzzle_pilot import save
from dynamic_skill.campaign_budget import CampaignBudget


def configure():
    base.ROOT = Path('results/subskill_exposure_20261007')
    base.AUTH = 'subskill_exposure_20261007_authorization'
    base.configure()
    base.trial.ARMS = {'X_exposure': {'macro_mode': True, 'discovery_interrupt': True}}


def prepare():
    configure()
    gate = read(Path('results/protocol_diagnostic_v5_20261007/diagnostic_audit.json'))
    assert gate['protocol_gate_passed'], 'Protocol diagnostic must pass before task expansion'
    cfg = read(Path('results/protocol_diagnostic_v5_20261007/config.json'))
    candidates = sorted(Path('/opt/data/private/data/alfworld/json_2.1.1/valid_seen').glob('*/*/game.tw-pddl'))
    used = set()
    for path in Path('results').rglob('config.json'):
        if path == base.ROOT / 'config.json':
            continue
        try:
            text = json.dumps(read(path))
        except (ValueError, OSError):
            continue
        used.update(str(p) for p in candidates if str(p) in text)
    tasks = []
    for kind in ('pick_and_place_simple', 'look_at_obj_in_light'):
        objects = set()
        for game in candidates:
            name = game.parent.parent.name
            if not name.startswith(kind + '-') or str(game) in used:
                continue
            object_type = name.split('-')[1]
            if object_type in objects:
                continue
            objects.add(object_type)
            tasks.append({'id': f'exposure-{kind}-{len(objects)}', 'gamefile': str(game),
                          'game_sha256': digest(game), 'benchmark': 'alfworld'})
            if len(objects) == 3:
                break
        assert len(objects) == 3
    cfg.update(protocol='subskills-v5-new-development-exposure', tasks=tasks,
               arms=base.trial.ARMS, job_order=[[i, 'X_exposure'] for i in range(6)])
    sources = sorted(set(cfg['source_hashes']) | {'experiments/subskill_exposure_trial.py',
        'experiments/audit_protocol_diagnostic.py'})
    cfg['source_hashes'] = {p: digest(Path(p)) for p in sources}
    cfg['note'] = ('Six new development tasks, three distinct object types per supported task family; '
        'deterministic path-only selection, excluding historical config paths. No answer/outcome selection. '
        'One arm measures exposure only; interrupt efficacy requires a later paired ablation. '
        'Same v5 prompt/model/settings/skill; discovery interrupt enabled, boundary prompt unchanged.')
    if (base.ROOT / 'config.json').exists():
        assert read(base.ROOT / 'config.json') == cfg
    else:
        save(base.ROOT / 'config.json', cfg)
        for path in sources:
            dst = base.ROOT / 'source_snapshot' / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
    print('PREPARED', config_hash(cfg))


def authorize():
    configure()
    cfg = read(base.ROOT / 'config.json')
    with (base.trial.CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = CampaignBudget(base.trial.CAMPAIGN / 'budget.json', deadline=0)
        data = budget.read()
        assert base.AUTH not in data and not data['policy_violation']
        assert not any(r['status'] == 'pending' for r in data['requests'])
        assert time.time() >= data['cooldown_until'] and data['limit_usd'] == 85
        assert all(digest(Path(p)) == h for p, h in cfg['source_hashes'].items())
        save(base.ROOT / 'budget_before_authorization.json', data)
        auth = {'config_sha256': config_hash(cfg), 'deadline_unix': time.time() + 3600,
            'max_requests': len(data['requests']) + 72, 'start_request_count': len(data['requests']),
            'batch_spend_ceiling': min(85, budget.effective_spend(data) + 5),
            'user_authorization': '2026-10-07 continue autonomously; six new development exposure tasks after protocol gate, cumulative USD85 unchanged'}
        data[base.AUTH] = auth
        data.update(deadline_unix=auth['deadline_unix'], max_requests=auth['max_requests'])
        budget.write(data)
        print('AUTHORIZED', json.dumps(auth))


def advance():
    configure()
    if not os.environ.get('TARGET_AZURE_OPENAI_API_KEY'):
        os.environ['TARGET_AZURE_OPENAI_API_KEY'] = Path('/opt/data/private/api-key').read_text().strip()
    os.environ.setdefault('ALFWORLD_DATA', '/opt/data/private/data/alfworld')
    base.trial.advance()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'authorize', 'summary'])
    args = parser.parse_args()
    configure()
    {'prepare': prepare, 'authorize': authorize, 'summary': base.trial.summary}[args.action]()

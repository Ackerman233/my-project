"""Freeze, inspect and run joint evolution; demo makes zero API requests."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.alfworld.common.alfworld_campaign import CAMPAIGN
from experiments.common.bounded_campaign import (freeze, authorize, check_frozen, open_batch,
    StageBudget, batch_accounting, auth_key, AUTHORIZED_LIMIT_USD)
from experiments.common.puzzle_pilot import Settings, save
from .schema import Genome, validate_splits
from .engine import evolve
from .skillopt_core import PIPELINE, MAX_CALLS, MAX_EDITS

PROJECT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exposed_scenes(scene_ids, *, exclude_root=None):
    """Conservative scene-level audit, also catching renamed paths and trials."""
    remaining, exposed = set(scene_ids), set()
    exclusions = exclude_root if isinstance(exclude_root, (list, tuple)) else [exclude_root] if exclude_root is not None else []
    for path in Path('results').rglob('*.json'):
        if 'source_snapshot' in path.parts or any(path.resolve().is_relative_to(Path(root).resolve()) for root in exclusions):
            continue
        text = path.read_text()
        found = {scene for scene in remaining if scene in text}
        exposed.update(found)
        remaining.difference_update(found)
        if not remaining:
            break
    return exposed


def validate_config(cfg, *, check_exposure=True, exclude_root=None):
    cfg = json.loads(json.dumps(cfg))
    generations = cfg.get('generations', 2)
    candidates = cfg.get('candidates_per_generation', 2)
    if type(generations) is not int or not 1 <= generations <= 20 or type(candidates) is not int or not 1 <= candidates <= 4:
        raise ValueError('Invalid generation/candidate limit')
    validate_splits(cfg['splits'], generations)
    initial = Genome.from_dict(cfg['initial'])
    settings = Settings(**cfg['settings'])
    if not (all(type(getattr(settings, k)) is int for k in ('max_calls', 'max_steps', 'max_tokens')) and
            1 <= settings.max_calls <= 64 and 1 <= settings.max_steps <= 256 and
            128 <= settings.max_tokens <= 2048 and settings.temperature == 0 and
            1 <= settings.request_timeout <= 120 and 1 <= settings.episode_seconds <= 1800):
        raise ValueError('Invalid frozen actor settings')
    if type(cfg.get('optimizer_max_tokens', 1536)) is not int or not 256 <= cfg.get('optimizer_max_tokens', 1536) <= 4096:
        raise ValueError('Invalid optimizer token cap')
    groups = [cfg['splits']['train'], cfg['splits']['selection'], *cfg['splits']['confirmation'], cfg['splits']['test']]
    if type(cfg.get('allow_previously_exposed_evaluation', False)) is not bool:
        raise ValueError('Exposure override must be an explicit boolean')
    hashes = set()
    for tasks in groups:
        for task in tasks:
            game = Path(task['gamefile']).resolve()
            if not game.is_file() or game.name != 'game.tw-pddl':
                raise ValueError('Require a real game.tw-pddl for live execution')
            if task['scene_id'] != game.parent.parent.name:
                raise ValueError('scene_id must equal physical game scene directory')
            split = game.parent.parent.parent.name
            if split not in ('train', 'valid_seen', 'valid_unseen'):
                raise ValueError('Unknown dataset split')
            sha = digest(game)
            if sha in hashes:
                raise ValueError('Duplicate physical game content across data splits')
            if task.get('game_sha256', sha) != sha:
                raise ValueError('Gamefile hash mismatch')
            hashes.add(sha)
            task.update(gamefile=str(game), game_sha256=sha, dataset_split=split, benchmark='alfworld')
    if check_exposure:
        evaluation = [*cfg['splits']['test'], *(t for fold in cfg['splits']['confirmation'] for t in fold)]
        if cfg.get('evaluation_reservation_source'):
            from .resume import unused_evaluation_reservation
            reserved=unused_evaluation_reservation(cfg)
            exposed=sorted(exposed_scenes({t['scene_id'] for t in evaluation},exclude_root=reserved))
        elif cfg.get('resume_parent'):
            from .resume import reservation_ancestors
            ancestors = reservation_ancestors(cfg)
            exposed = sorted(exposed_scenes({t['scene_id'] for t in evaluation}, exclude_root=ancestors))
        else:
            exposed = sorted(exposed_scenes({t['scene_id'] for t in evaluation}, exclude_root=exclude_root))
        if exposed and not cfg.get('allow_previously_exposed_evaluation', False):
            raise ValueError('Confirmation/test scenes have prior exposure; select new scenes or explicitly label development-only')
        cfg['exposure_audit'] = {'previously_exposed_evaluation_scenes': exposed,
                                 'independent_evaluation_claim_allowed': not exposed}
    cfg.update(initial=asdict(initial), generations=generations, candidates_per_generation=candidates,
               settings=asdict(settings), optimizer_max_tokens=cfg.get('optimizer_max_tokens', 1536))
    optimizer = {'pipeline': PIPELINE, 'max_calls_per_candidate': MAX_CALLS,
                 'max_selected_edits': MAX_EDITS, 'meta_skill_enabled': False}
    if cfg.get('optimizer', optimizer) != optimizer:
        raise ValueError('Unsupported optimizer configuration')
    cfg['optimizer'] = optimizer
    if cfg.get('citation_mode','per_edit') not in ('per_edit','batch_provenance'):raise ValueError('Invalid citation mode')
    for flag in ('footer_contract','test_unchanged','pooled_transport'):
        if type(cfg.get(flag,True)) is not bool:raise ValueError('Explicit '+flag+' boolean required')
    if cfg.get('network_route', 'environment') not in ('environment', 'direct'):
        raise ValueError('Unknown network route')
    if cfg.get('resume_parent') or cfg.get('training_source'):
        from .resume import cache_manifest
        manifest = cache_manifest(cfg)
        if cfg.get('training_cache', manifest) != manifest:
            raise ValueError('Frozen training cache changed')
        cfg['training_cache'] = manifest
        from .resume import branch_cache_manifest
        branches=branch_cache_manifest(cfg)
        if cfg.get('branch_cache',branches)!=branches:
            raise ValueError('Frozen branch cache changed')
        cfg['branch_cache']=branches
    if cfg.get('branch_feedback', False) not in (True, False) or type(cfg.get('connection_retry', False)) is not bool:
        raise ValueError('Explicit feedback/retry booleans required')
    scopes = cfg.get('candidate_scopes')
    if scopes is not None and (not isinstance(scopes, list) or len(scopes) != candidates or
                              any(s not in ('any', 'skill_only', 'policy_or_joint') for s in scopes)):
        raise ValueError('One bounded scope per candidate required')
    if cfg.get('cumulative_limit_usd', AUTHORIZED_LIMIT_USD) != AUTHORIZED_LIMIT_USD:
        raise ValueError('Cumulative cap must match the current explicit user authorization')
    cfg['cumulative_limit_usd'] = AUTHORIZED_LIMIT_USD
    return cfg


def prepare(config_path, root):
    cfg = validate_config(json.loads(Path(config_path).read_text()))
    if not (CAMPAIGN / 'budget.json').is_file():
        raise ValueError('Existing shared campaign ledger required')
    budget = StageBudget(root)
    ledger = budget.read()
    if ledger['limit_usd'] != AUTHORIZED_LIMIT_USD:
        raise ValueError('Shared ledger must match the explicitly authorized cumulative cap')
    available = AUTHORIZED_LIMIT_USD - budget.effective_spend(ledger)
    batch = cfg.get('batch_usd', min(5.0, available))
    if type(batch) not in (int, float) or not 0 < batch <= available:
        raise ValueError('Batch cap exceeds remaining cumulative allowance')
    g, c, splits = cfg['generations'], cfg['candidates_per_generation'], cfg['splits']
    episodes = sum(len(splits['train']) + (c + 1) * len(splits['selection']) + 2 * len(fold)
                   for fold in splits['confirmation']) + 4 * len(splits['test'])
    branch_requests = g * len(splits['train']) * 2 * 4 if cfg.get('branch_feedback') else 0
    retry_factor = 2 if cfg.get('connection_retry') else 1
    cfg.update(protocol='joint-skill-interrupt-live-v3', limits={'batch_usd': batch,
        'max_requests': retry_factor * (episodes * cfg['settings']['max_calls'] + g * c * MAX_CALLS + branch_requests),
        'max_new_unknown': 4 if cfg.get('connection_retry') else 1, 'seconds': 14400})
    files = list(Path('dynamic_skill').rglob('*.py')) + list(Path('experiments').rglob('*.py'))
    files += [Path('dynamic_skill/migration_20261008.json'), Path('experiments/migration_20261008.json'),
              Path('skillopt/skillopt/optimizer/skill.py')]
    # Freeze the upstream algorithms AND their loaded prompts, not only Apply.
    for part in ('gradient', 'optimizer', 'prompts', 'utils'):
        files += list(Path('skillopt/skillopt', part).rglob('*.py'))
    files += list(Path('skillopt/skillopt/prompts').glob('*.md'))
    files += list(Path('skillopt/skillopt/envs/alfworld/prompts').glob('*.md'))
    files += [p for p in (Path('skillopt/skillopt/utils.py'),) if p.exists()]
    cfg['source_hashes'] = {str(p): digest(p) for p in sorted(set(files)) if '__pycache__' not in p.parts}
    freeze(root, cfg)
    print(json.dumps({'prepared': str(root), 'maximum_episodes': episodes, 'maximum_requests': cfg['limits']['max_requests'],
                      'batch_usd': batch, 'paid_requests': 0, 'authorization_opened': False}))


def preflight(root):
    cfg = json.loads((root / 'config.json').read_text())
    if cfg.get('protocol') != 'joint-skill-interrupt-live-v3' or cfg.get('optimizer', {}).get('pipeline') != PIPELINE:
        raise ValueError('Prepare a new frozen v3 batch for the SkillOpt core pipeline')
    check_frozen(root, cfg)
    validate_config(cfg, check_exposure=False)
    if not (CAMPAIGN / 'budget.json').is_file():
        raise ValueError('Existing shared ledger required')
    budget = StageBudget(root)
    state = budget.read()
    auth = state.get(auth_key(root), {})
    ready = bool(auth and not auth.get('closed') and time.time() < auth.get('deadline_unix', 0) and
                 len(state['requests']) < state['max_requests'] and not state['policy_violation'] and
                 not any(r['status'] == 'pending' for r in state['requests']))
    return {'source_and_data_valid': True, 'authorization_open': ready,
            'cumulative_usd': budget.effective_spend(state), 'limit_usd': state['limit_usd'],
            'paid_requests': 0, 'full_run_affordability_not_guaranteed': True}


def run(root):
    from .runtime import LiveTransport
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not preflight(root)['authorization_open']:
            raise ValueError('Stage authorization is closed; no request made')
        if (root / 'run').exists():
            raise ValueError('Prior run exists; no implicit overwrite or retry')
        cfg, budget = open_batch(root)
        transport = LiveTransport(root, budget, Settings(**cfg['settings']),
            connection_retry=cfg.get('connection_retry', False), candidate_scopes=cfg.get('candidate_scopes'),
            network_route=cfg.get('network_route', 'environment'), training_cache=cfg.get('training_cache'), branch_cache=cfg.get('branch_cache'), footer_contract=cfg.get('footer_contract',False), pooled_transport=cfg.get('pooled_transport',False), citation_mode=cfg.get('citation_mode','per_edit'))
        try:
            evolve(cfg['splits'], Genome.from_dict(cfg['initial']), transport.evaluator,
                   None, root / 'run', generations=cfg['generations'],
                   candidates_per_generation=cfg['candidates_per_generation'],
                   candidate_builder=transport.candidate_builder(cfg['optimizer_max_tokens']),
                   evidence_enricher=transport.evidence_enricher if cfg.get('branch_feedback') else None,
                   test_unchanged=cfg.get('test_unchanged',True))
        finally:
            transport.close()
            save(root / 'accounting.json', batch_accounting(root))
            state = budget.read()
            if not any(r['status'] == 'pending' for r in state['requests']):
                state[auth_key(root)]['closed'] = True
                state['max_requests'] = len(state['requests'])
                state['deadline_unix'] = time.time()
                budget.write(state)
            save(root / 'budget_after.json', budget.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    demo = sub.add_parser('demo', help='Offline upstream SkillOpt core fixture, no benchmark gain claim')
    demo.add_argument('--out', type=Path, default=Path('results/alfworld/joint_evolution/20261008_skillopt_core_demo'))
    core_demo = sub.add_parser('skillopt-demo', help='Offline fixture running upstream Reflect/Aggregate/Rank/Apply')
    core_demo.add_argument('--out', type=Path, default=Path('results/alfworld/joint_evolution/20261008_skillopt_core_demo'))
    prep = sub.add_parser('prepare', help='Validate splits and freeze sources; no paid requests')
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    for name in ('preflight', 'authorize', 'run'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    for name in ('out', 'root', 'config'):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    os.chdir(PROJECT)
    if args.action in ('demo', 'skillopt-demo'):
        from .skillopt_demo import run_demo
        best, _ = run_demo(args.out)
        print(json.dumps({'offline_demo_complete': True, 'best_revision': best.revision, 'paid_requests': 0, 'out': str(args.out)}))
    elif args.action == 'prepare':
        prepare(args.config, args.out)
    elif args.action == 'preflight':
        print(json.dumps(preflight(args.root)))
    elif args.action == 'authorize':
        preflight(args.root)
        authorize(args.root)
    else:
        run(args.root)


if __name__ == '__main__':
    main()

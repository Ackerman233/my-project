"""Frozen, fresh-scene rule/brief-judge/dependency-judge trial; no online prompt tuning."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from dynamic_skill.controllers.judge import JudgeConfig, PROMPTS, SYSTEM
from dynamic_skill.infrastructure.campaign_budget import CampaignHalt
from experiments.alfworld.common.alfworld_campaign import CAMPAIGN
from experiments.alfworld.joint_evolution.schema import Genome
from experiments.alfworld.joint_evolution.runtime import LiveTransport
from experiments.alfworld.joint_evolution.engine import CLEAN, paired_gate
from experiments.common.bounded_campaign import (AUTHORIZED_LIMIT_USD, StageBudget,
    freeze, check_frozen, authorize, open_batch, batch_accounting, auth_key)
from experiments.common.puzzle_pilot import Settings, save
from .runtime import evaluate

ARMS = ('rules', 'brief', 'dependency')
FAMILIES = ('pick_and_place_simple', 'pick_clean_then_place_in_recep',
            'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep',
            'look_at_obj_in_light', 'pick_two_obj_and_place')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_tasks(count=10):
    """Exclude ALL scene names mentioned in historical results/configs, even reservations.

    This is scene-directory independence, not guaranteed room-layout independence.
    Select by seeded shuffle/family round robin, never by outcome or opportunity.
    """
    pattern = re.compile(r'(?:' + '|'.join(FAMILIES) + r')-[A-Za-z0-9_-]+')
    exposed = set()
    for root in (Path('results'), Path('experiments')):
        for path in root.rglob('*.json'):
            if 'source_snapshot' not in path.parts:
                exposed.update(pattern.findall(path.read_text(errors='replace')))
    base = Path('/opt/data/private/data/alfworld/json_2.1.1/valid_seen')
    rng = random.Random(20261010)
    pools = {}
    for family in FAMILIES:
        scenes = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(family+'-') and p.name not in exposed)
        rng.shuffle(scenes)
        pools[family] = scenes
    groups, used_hashes = {}, set()
    for split in ('selection', 'confirmation'):
        tasks = []
        while len(tasks) < count:
            added = False
            for family in FAMILIES:
                if len(tasks) == count:
                    break
                while pools[family]:
                    scene = pools[family].pop()
                    games = sorted(scene.glob('*/game.tw-pddl'))
                    if not games:
                        continue
                    game = games[0]
                    sha = digest(game)
                    if sha in used_hashes:
                        continue
                    used_hashes.add(sha)
                    tasks.append({'id': f'judge200-{split}-{len(tasks):02d}',
                        'scene_id': scene.name, 'gamefile': str(game), 'game_sha256': sha,
                        'dataset_split': 'valid_seen', 'task_family': family})
                    added = True
                    break
            if not added:
                raise ValueError('Not enough fresh scenes')
        groups[split] = tasks
    return groups, {'seed': 20261010, 'previously_mentioned_scene_count': len(exposed),
                    'scene_directory_disjoint': True, 'room_layout_independence': 'not established',
                    'family_counts': {k: dict(Counter(t['task_family'] for t in v)) for k, v in groups.items()}}


def prepare(root):
    source = Path('experiments/alfworld/joint_evolution/configs/20261008_paid160_provenance.json')
    old = json.loads(source.read_text())
    splits, audit = fresh_tasks()
    cfg = {'protocol': 'gated-local-judge-v2-hard-evidence', 'cumulative_limit_usd': AUTHORIZED_LIMIT_USD,
        'initial': old['initial'], 'settings': {**old['settings'], 'max_tokens': 1024,
            'max_calls': 8, 'max_steps': 40, 'episode_seconds': 900},
        'judge_model': old['settings']['model'], 'judge_config': asdict(JudgeConfig(authoritative_admissibility=True)),
        'prompts': {'system': SYSTEM, **PROMPTS}, 'arms': list(ARMS), 'splits': splits,
        'exposure_audit': audit, 'network_route': 'environment', 'pooled_transport': True,
        'connection_retry': False, 'limits': {'batch_usd': 48, 'max_requests': 400,
            'max_new_unknown': 2, 'seconds': 14400},
        'preregistration': {
            'comparison': 'Fresh 10 tasks x 3 arms. Rotate arm order within each task. Same skill, model, primitive actions, seed42 and actor prompt. Actor output cap1024 in every arm; older512-cap results are not a paired comparator.',
            'judge_gate': 'Only nonterminal nonempty queues. ALL ARMS first interrupt without a judge call if the NEXT command is absent from the complete public ALFWorld action list. Only otherwise query for a rule anomaly or new publicly admissible take command. Checking later actions against current admissibility is forbidden. This opportunity adapter is ALFWorld-specific; the dependency prompt is generic.',
            'budget': 'Actor+judge share8 total requests per episode,40 actions. Judge max4 requests,128 output tokens; preserve one remaining actor call. No generation retries. All actual actor/judge charges and unknown reservations retained.',
            'fallback': 'Invalid JSON/truncation/defer preserves baseline rule and records paid cost. Infrastructure/budget errors halt batch. Terminal status and empty queues never sent to judge.',
            'acceptance': 'Existing paired_gate on TOTAL calls and total cost; no successful task lost. Candidate with judge format errors cannot pass. Best passing arm alone faces10 fresh confirmation tasks against rules. No post-outcome prompt change; no acceptance on selection alone.',
            'scope': 'Fixed prompt role ablation, not learned judge, full SkillOpt training, cheaper model evidence, or final benchmark test. If neither passes selection, skip confirmation. Counterfactual attribution requires separately replaying the same prefix; full-episode paired differences alone do not prove a specific interrupt caused a win.',
        }}
    paths = sorted([*Path('dynamic_skill').rglob('*.py'), *Path('experiments').rglob('*.py')])
    paths += sorted(Path('skillopt/skillopt').rglob('*.py'))
    cfg['source_hashes'] = {str(p): digest(p) for p in paths}
    cfg['input_hashes'] = {str(source): digest(source)}
    for tasks in splits.values():
        for t in tasks:
            game = Path(t['gamefile'])
            for p in game.parent.iterdir():
                if p.is_file():
                    cfg['input_hashes'][str(p)] = digest(p)
    budget = StageBudget(root)
    if budget.read()['limit_usd'] != AUTHORIZED_LIMIT_USD:
        raise ValueError('Raise authorized cumulative cap first')
    if cfg['limits']['batch_usd'] > AUTHORIZED_LIMIT_USD - budget.effective_spend(budget.read()):
        raise ValueError('Insufficient remaining allowance for configured batch')
    freeze(root, cfg)
    save(root / 'prepared.json', {'paid_requests': 0, 'tasks': 20, 'selection_episodes': 30,
                                 'confirmation_episodes_if_selected': 20})


def summarize(rows, cfg):
    totals = {}
    for arm in ARMS:
        values = [r for r in rows if r['arm'] == arm]
        totals[arm] = {'episodes': len(values), 'successes': sum(r['success'] for r in values),
                      'statuses': dict(Counter(r['status'] for r in values))}
        for key in ('actor_calls', 'judge_calls', 'total_model_calls', 'cost_usd', 'actor_cost_usd',
                    'judge_cost_usd', 'environment_steps', 'invalid_actions', 'rule_overrides',
                    'actual_tail_interruptions', 'judge_format_errors', 'judge_latency_seconds'):
            totals[arm][key] = sum(r[key] for r in values)
    baseline = [r for r in rows if r['arm'] == 'rules']
    gates = {}
    for arm in ARMS[1:]:
        candidate = [r for r in rows if r['arm'] == arm]
        gate = paired_gate(baseline, candidate)
        if len(baseline) != len(cfg['splits']['selection']):
            gate = {'accept': False, 'reason': 'incomplete_scheduled_tasks'}
        if any(r['judge_format_errors'] for r in candidate):
            gate = {**gate, 'accept': False, 'reason': 'judge_format_error'}
        gates[arm] = gate
    return {'totals': totals, 'gates': gates}


def audit_initials(root, rows):
    by_task = {}
    for row in rows:
        path = root / 'run' / row['split'] / row['task_id'] / row['arm'] / 'predictions' / row['id'] / 'initial.json'
        by_task.setdefault(row['task_id'], {})[row['arm']] = digest(path)
    return {'tasks': by_task, 'all_available_pairs_identical': all(len(set(d.values())) == 1 for d in by_task.values())}


def run(root):
    with (CAMPAIGN / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, budget = open_batch(root)
        if (root / 'run').exists():
            raise ValueError('Do not overwrite or silently rerun a paid trial')
        (root / 'run').mkdir()
        settings = Settings(**cfg['settings'])
        transport = LiveTransport(root, budget, settings, pooled_transport=cfg['pooled_transport'],
            connection_retry=False, network_route=cfg['network_route'])
        judge_transport = transport if cfg['judge_model'] == settings.model else LiveTransport(
            root, budget, replace(settings, model=cfg['judge_model']), pooled_transport=cfg['pooled_transport'],
            connection_retry=False, network_route=cfg['network_route'])
        genome = Genome.from_dict(cfg['initial'])
        rows, completion = [], {'status': 'running', 'accepted': False}
        try:
            for split in ('selection', 'confirmation'):
                selected = completion.get('selected')
                if split == 'confirmation' and not selected:
                    save(root / 'run/confirmation_skipped.json', {'reason': 'No selection gain'})
                    break
                arms = ARMS if split == 'selection' else ('rules', selected)
                for i, task in enumerate(cfg['splits'][split]):
                    for arm in arms[i % len(arms):] + arms[:i % len(arms)]:
                        out = root / 'run' / split / task['id'] / arm
                        env = transport.environment(task['gamefile'])
                        try:
                            result = evaluate(env, task, genome, transport, out, arm,
                                              cfg['judge_config'], judge_transport)
                        finally:
                            env.close()
                        result['split'] = split
                        rows.append(result)
                        save(root / 'run/results.json', rows)
                        save(root / 'run/paired_initials.json', audit_initials(root, rows))
                        if result['status'] not in CLEAN | {'format_error'}:
                            raise CampaignHalt(result['status'])
                summary = summarize([r for r in rows if r['split'] == split], cfg)
                save(root / 'run' / split / 'comparison.json', summary)
                if split == 'selection':
                    eligible = [a for a in ARMS[1:] if summary['gates'][a]['accept']]
                    eligible.sort(key=lambda a: (-summary['totals'][a]['successes'],
                        summary['totals'][a]['total_model_calls'], summary['totals'][a]['cost_usd'], a))
                    completion['selected'] = eligible[0] if eligible else None
                    save(root / 'run/frozen_before_confirmation.json', completion)
                else:
                    completion['accepted'] = summary['gates'][selected]['accept']
            completion['status'] = 'complete'
        except CampaignHalt as exc:
            completion.update(status='stopped', reason=exc.status)
        except Exception as exc:
            completion.update(status='error', error_type=type(exc).__name__)
            raise
        finally:
            transport.close()
            if judge_transport is not transport:
                judge_transport.close()
            completion['completed_episodes'] = len(rows)
            save(root / 'completion.json', completion)
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
    parser.add_argument('action', choices=('prepare', 'authorize', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    {'prepare': prepare, 'authorize': authorize, 'run': run}[args.action](root)


if __name__ == '__main__':
    main()

"""Audit frozen three-arm trial, separating coverage, mechanics, and task benefit."""
from pathlib import Path
from math import comb
import json
from experiments.interrupt_recovery_ablation import read, digest, config_hash
from experiments.puzzle_pilot import save
from experiments.queue_revision_transfer_trial import ROOT, AUTH, ARMS, RULES
from experiments.alfworld_campaign import CAMPAIGN
from dynamic_skill.campaign_budget import CampaignBudget
from dynamic_skill.queue_revision import revise_head, Truth
from dynamic_skill.queue_evidence import AlfworldEvidence


def discordance(pairs, a, b):
    wins = sum(p['arms'][a]['success'] and not p['arms'][b]['success'] for p in pairs)
    losses = sum(p['arms'][b]['success'] and not p['arms'][a]['success'] for p in pairs)
    n = wins + losses
    p = min(1., 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.
    return {'pairs': len(pairs), 'wins': wins, 'losses': losses, 'ties': len(pairs)-n,
            'two_sided_exact_discordance_p': p,
            'note': 'Exploratory small-sample paired comparison, not a confirmatory powered test.'}


def main():
    cfg, frozen, summary = (read(ROOT / name) for name in ('config.json', 'cases.json', 'summary.json'))
    assert config_hash(cfg) == frozen['config_sha256']
    assert all(digest(Path(p)) == h for p, h in cfg['source_hashes'].items())
    assert all(digest(ROOT / 'source_snapshot' / p) == h for p, h in cfg['source_hashes'].items())
    previous = read(Path('results/queue_revision_principles_20261007/source_hashes.json'))
    assert all(previous[p] == cfg['rule_hashes'][p] for p in RULES)
    ledger = CampaignBudget(CAMPAIGN / 'budget.json', deadline=0)
    state = ledger.read()
    auth = state[AUTH]
    assert digest(ROOT / 'cases.json') == auth['cases_sha256']
    requests = [r for r in state['requests'][auth['start_request_count']:] if r['label'].startswith(str(ROOT) + '/')]
    assert not any(r['status'] == 'pending' for r in requests)
    full = [p for p in summary['pairs'] if p['complete']]
    assert len(full) == len(frozen['cases']) and all(p['checkpoint_identical'] for p in full)
    for case in frozen['cases']:
        assert digest(Path(case['source_trace'])) == case['source_sha256']
        assert digest(Path(case['source_trace']).parent / 'initial.json') == case['source_initial_sha256']
        for arm in ARMS:
            out = ROOT / 'online' / case['id'] / arm
            assert config_hash(read(out / 'checkpoint.json')) == case['checkpoint_sha256']
            result, trace, calls = read(out / 'result.json'), read(out / 'suffix_trace.json') if (out / 'suffix_trace.json').exists() else [], read(out / 'model_calls.json') if (out / 'model_calls.json').exists() else []
            assert len(trace) == result['suffix_steps'] <= cfg['settings']['max_steps']
            assert len(calls) == result['model_calls'] <= cfg['settings']['max_calls']
            assert abs(result['cost_usd'] - sum(r['budget_charge_usd'] for r in requests if r['label'] == str(out))) < 1e-8
    protocol_ok = [p for p in full if all(r['status'] not in ('format_error', 'api_error', 'unknown_cost_stop', 'thinking_policy_stop') for r in p['arms'].values())]
    collection = []
    shadow = []
    for task in cfg['tasks']:
        directory = ROOT / 'collection' / 'predictions' / (cfg['model'] + '__' + task['id'] + '__fixed_chunk')
        result = read(directory / 'result.json')
        collection.append({'task_id': task['id'], 'object_type': task['object_type'], 'family': task['family'],
            'status': result['status'], 'model_calls': result['model_calls'], 'steps': result['environment_steps'], 'cost_usd': result['cost_usd']})
        trace = read(directory / 'conversation.json') if (directory / 'conversation.json').exists() else []
        for index, row in enumerate(trace[:-1]):
            tail = row.get('remaining_before_interrupt', [])
            if row['done'] or not tail:
                continue
            admissible = trace[index+1]['admissible_before']
            revision = revise_head(tail, AlfworldEvidence(row['env_feedback'], admissible, complete=False))
            shadow.append({'task_id':task['id'], 'step':row['step'], 'tail':tail,
                'head_verdict':revision.proofs[0].value.value,
                'decision':revision.decision, 'next_action':revision.next_action,
                'actual_following_action':trace[index+1]['action'],
                'actual_following_valid':trace[index+1]['valid'],
                'reason':revision.reason})

    coverage = []
    for case in frozen['cases']:
        cp = case['checkpoint']
        revision = revise_head(case['tail'], AlfworldEvidence(cp['observation'], cp['admissible'], complete=False))
        repair = read(ROOT / 'online' / case['id'] / 'evidence_repair' / 'result.json')
        coverage.append({'case_id': case['id'], 'kind': case['kind'], 'prefix_steps': cp['prefix_steps'],
            'tail': case['tail'], 'initial_decision': revision.decision, 'initial_skipped': list(revision.skipped),
            'initial_retained_next': revision.next_action, 'repair_original_actions_executed': repair['original_tail_executed'],
            'repair_revision_replan_discarded': repair['revision_replan_discarded']})
    report = {'config_sha256': config_hash(cfg), 'cases_sha256': digest(ROOT / 'cases.json'),
        'rules_unchanged_from_principle_validation': True, 'source_and_checkpoint_checks_passed': True,
        'tasks': len(cfg['tasks']), 'source_task_count': cfg.get('source_task_count', len(cfg['tasks'])), 'prefixes': len(full), 'protocol_valid_triplets': len(protocol_ok),
        'falsified_prefixes': sum(c['kind'] == 'falsified_head' for c in coverage),
        'immediate_skip_and_retain_prefixes': sum(c['initial_decision'] == 'skip_head' for c in coverage),
        'untriggered_controls': sum(c['kind'] == 'untriggered_control' for c in coverage),
        'coverage': coverage, 'collection': collection, 'arms': summary['arms'],
        'collection_shadow': {'evaluated_nonempty_checkpoints':len(shadow),
            'falsified_heads':sum(r['head_verdict']=='false' for r in shadow),
            'unknown_heads':sum(r['head_verdict']=='unknown' for r in shadow),
            'skip_and_retain':sum(r['decision']=='skip_head' for r in shadow),
            'rows':shadow, 'scope':'Offline shadow decisions at all collection queues with next-step public commands; not new paired interventions.'},
        'paired_operational': {b: discordance(full, 'evidence_repair', b) for b in ARMS[:2]},
        'paired_protocol_valid': {b: discordance(protocol_ok, 'evidence_repair', b) for b in ARMS[:2]},
        'falsified_subset': {'pairs': sum(p['kind']=='falsified_head' for p in full),
            'arms': {a: {'successes': sum(p['arms'][a]['success'] for p in full if p['kind']=='falsified_head'),
                **{k: sum(p['arms'][a][k] for p in full if p['kind']=='falsified_head') for k in ('model_calls','suffix_steps','invalid_actions','cost_usd')}} for a in ARMS}},
        'batch': {'requests': len(requests), 'known_usage': sum(r['status']=='returned_usage' for r in requests),
            'cost_usd': sum(r['budget_charge_usd'] for r in requests), 'cumulative_requests': len(state['requests']),
            'cumulative_usd': ledger.effective_spend(state), 'pending': sum(r['status']=='pending' for r in state['requests'])},
        'scope': cfg['note'],
        'audit_source_sha256': digest(Path(__file__))}
    save(ROOT / 'transfer_audit.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('coverage','collection','collection_shadow','scope','audit_source_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

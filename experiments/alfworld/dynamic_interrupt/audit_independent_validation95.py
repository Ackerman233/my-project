"""Source, independence, checkpoint, cost and gate audit. Never calls a model."""
import json
from pathlib import Path
from experiments.unknown_independent_validation import ROOT,ARMS,BAD,summary,used_scene_names
from experiments.bounded_research95 import StageBudget,auth_key,check_frozen,batch_accounting
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.puzzle_pilot import save


def main():
    cfg=read(ROOT/'config.json');frozen=read(ROOT/'cases.json');report=summary()
    check_frozen(ROOT,cfg)
    assert report['complete'] and all(p['checkpoint_identical'] for p in report['pairs'])
    previous=read(Path('results/queue_revision_principles_20261007/source_hashes.json'))
    assert all(previous[p]==sha for p,sha in cfg['rule_hashes'].items())
    games=[Path(t['gamefile']) for t in cfg['tasks']]
    assert all('valid_unseen' in game.parts for game in games)
    assert all(digest(Path(t['gamefile']))==t['game_sha256'] for t in cfg['tasks'])
    catalog_path=ROOT/'independence_history_catalog.json'
    if not catalog_path.exists():
        historical=[p for p in Path('results').rglob('*.json') if ROOT not in p.parents and 'source_snapshot' not in p.parts and p.name in ('config.json','initial.json','tasks.json')]
        save(catalog_path,{'historical_inputs':{str(p):digest(p) for p in historical},'note':'Catalog audited before subsequent paper-method trials; later trials must not retroactively change independence.'})
    catalog=read(catalog_path)['historical_inputs']
    assert all(digest(Path(p))==sha for p,sha in catalog.items())
    used=used_scene_names([Path(p) for p in catalog],games)
    assert not {game.parent.parent.name for game in games}&used
    budget=StageBudget(ROOT);data=budget.read();auth=data[auth_key(ROOT)]
    assert auth['config_sha256']==config_hash(cfg) and auth['cases_sha256']==digest(ROOT/'cases.json')
    requests=data['requests'][auth['start_request_count']:]
    for case in frozen['cases']:
        assert digest(Path(case['source_trace']))==case['source_sha256']
        for arm in ARMS:
            out=ROOT/'online'/case['id']/arm
            result=read(out/'result.json')
            assert config_hash(read(out/'checkpoint.json'))==case['checkpoint_sha256']
            calls=read(out/'model_calls.json') if (out/'model_calls.json').exists() else []
            assert len(calls)==result['model_calls']<=cfg['settings']['max_calls']
            assert result['suffix_steps']<=cfg['settings']['max_steps']
            assert abs(sum(r['budget_charge_usd'] for r in requests if r['label']==str(out))-result['cost_usd'])<1e-9
            for call in calls:
                assert 'publicly observed queue/state contradiction' not in call['prompt']
    proof={'rule_hashes_unchanged':True,'new_unseen_task_scenes_verified':True,'sources_and_snapshots_verified':True,
        'identical_public_checkpoints':True,'same_suffix_budgets':True,'costs_match_ledger':True,
        'unknown_language_not_mislabeled_contradiction':True,'config_sha256':config_hash(cfg),
        'cases_sha256':digest(ROOT/'cases.json'),'summary_sha256':digest(ROOT/'summary.json'),
        'gain_gate_result':report['task_gain_demonstrated'],'audit_source_sha256':digest(Path(__file__)),
        'batch':batch_accounting(ROOT)}
    save(ROOT/'validation_audit.json',proof)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    main()

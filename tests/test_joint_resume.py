from dataclasses import asdict
import json
from pathlib import Path
import pytest
from experiments.alfworld.joint_evolution.resume import reuse_training, digest, validate_parent
from experiments.alfworld.joint_evolution.schema import Genome
from experiments.common.puzzle_pilot import save


def test_training_reuse_keeps_all_outcomes_and_checks_hashes(tmp_path):
    source=tmp_path/'old';source.mkdir();g=Genome('Skill.')
    row={'id':'model__task','task_id':'task','status':'model_call_limit','success':False,'genome_hash':g.hash,'cost_usd':.3}
    save(source/'result.json',row);save(source/'conversation.json',[{'action':'look'}])
    cache={'task':{'result_path':str(source/'result.json'),'file_hashes':{str(p):digest(p) for p in source.glob('*.json')}}}
    out=tmp_path/'new/execution_feedback'
    reused=reuse_training(cache,{'id':'task'},g,out)
    assert not reused['success'] and reused['cost_usd']==.3 and reused['new_paid_requests']==0
    assert reuse_training(cache,{'id':'task'},g,tmp_path/'selection') is None
    (source/'conversation.json').write_text('[]')
    with pytest.raises(ValueError,match='changed'):reuse_training(cache,{'id':'task'},g,tmp_path/'other/execution_feedback')


def test_evaluation_reservation_can_only_be_inherited_before_any_selection(tmp_path):
    parent={'exposure_audit':{'independent_evaluation_claim_allowed':True}}
    for key in ('splits','settings','initial','generations','candidates_per_generation','candidate_scopes','branch_feedback','optimizer_max_tokens','optimizer'):parent[key]={}
    parent['initial'] = {'evidence_ids': []}
    save(tmp_path/'config.json',parent);save(tmp_path/'run/status.json',{'status':'stopped'})
    (tmp_path/'run/generations/000').mkdir(parents=True)
    cfg={**parent,'resume_parent':str(tmp_path),'resume_parent_config_sha256':digest(tmp_path/'config.json')}
    cfg['initial'] = {'evidence_ids': ()}
    assert validate_parent(cfg)==tmp_path
    (tmp_path/'run/generations/000/selection').mkdir()
    with pytest.raises(ValueError,match='already entered'):validate_parent(cfg)


def test_branch_cache_keeps_failures_and_rejects_changed_checkpoint(tmp_path):
    from experiments.alfworld.joint_evolution.resume import reuse_branch
    source=tmp_path/'old';source.mkdir()
    case={'id':'task','step':3,'tail':['go to desk 1'],'kind':'ordinary_nonempty_queue'}
    row={'case_id':'task','arm':'continue_tail','status':'model_call_limit','success':False,'cost_usd':.2}
    save(source/'result.json',row)
    cache={'task/continue_tail':{'result_path':str(source/'result.json'),'case_signature':case,
        'file_hashes':{str(source/'result.json'):digest(source/'result.json')}}}
    reused=reuse_branch(cache,case,'continue_tail',tmp_path/'new')
    assert not reused['success'] and reused['new_paid_requests']==0
    with pytest.raises(ValueError,match='checkpoint'):reuse_branch(cache,{**case,'step':4},'continue_tail',tmp_path/'bad')

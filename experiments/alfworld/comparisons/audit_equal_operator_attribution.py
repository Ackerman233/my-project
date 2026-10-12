"""Check equal public operator privileges and isolate native queues/real interrupts."""
import argparse
import json
from pathlib import Path
from experiments.equal_operator_attribution import ROOT,ARMS,WORKFLOWS
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.bounded_research95 import check_frozen,batch_accounting
from experiments.paper_method_trial import BAD
from experiments.puzzle_pilot import save
from dynamic_skill.observable_search_operator import ObservableSearch


def actual_tail_interruption(row):
    return bool(not row['done'] and row.get('interruption_reason') and row['actual_discarded_actions'])


def audit(root=ROOT):
    cfg=read(root/'config.json');check_frozen(root,cfg)
    summary=read(root/'summary.json');assert summary['complete']
    assert len({json.dumps(s['workflow'],sort_keys=True) for s in cfg['arms'].values()})==1
    assert cfg['arms']['matched_fixed']['max_actions']==cfg['arms']['matched_dynamic']['max_actions']==4
    rows=[];physical={};prefixes={};invalid=[]
    for task in cfg['tasks']:
        physical.setdefault(task['physical_task_id'],[])
        for arm in cfg['arms']:
            out=root/'online'/task['id']/arm;r=read(out/'result.json');calls=read(out/'model_calls.json')
            trace=read(out/'conversation.json') if (out/'conversation.json').exists() else []
            initial=read(out/'initial.json');physical[task['physical_task_id']].append(config_hash(initial))
            assert r['model_calls']==len(calls)<=cfg['settings']['max_calls']
            assert r['environment_steps']==len(trace)<=cfg['settings']['max_steps']
            assert r['compiled_search_actions']==sum(x['action_origin']=='compiled_public_search' for x in trace)
            search=ObservableSearch(initial['task'],acquire=True,priority='surface_first')
            prefix=[];model_seen=False
            for x in trace:
                if x['action_origin']=='compiled_public_search':
                    assert x['action'] in x['admissible_before']
                    expected=search.action('',x['admissible_before'])
                    assert expected==x['operator'] and expected['actions']==[x['action']]
                    if not model_seen:prefix.append(x['action'])
                else:model_seen=True
                search.update(x['action'],x['env_feedback'],[],x['step'])
            assert search.public_state()['held']==r['search_public_state']['held']
            for call in calls:
                assert call['action_limit']<=cfg['arms'][arm]['max_actions']
                assert 'Shared public operator state: ' in call['prompt']
                assert not any(a.startswith('@') for a in call['actions'])
            # An API fault can occur before the first model action, but automatic prefix is still observable.
            prefixes.setdefault(task['physical_task_id'],[]).append(prefix)
            r={**r,'physical_task_id':task['physical_task_id'],'replicate':task['replicate'],
               'native_multi_calls':sum(len(c['actions'])>1 for c in calls),
               'actual_tail_interruptions':sum(actual_tail_interruption(x) for x in trace),
               'discarded_primitives':sum(len(x['actual_discarded_actions']) for x in trace if actual_tail_interruption(x)),
               'terminal_queue_cleanups':sum(bool(x['actual_discarded_actions']) and x['done'] for x in trace)}
            rows.append(r)
            if r['status'] in BAD:invalid.append({'task':task['id'],'arm':arm,'status':r['status']})
    assert all(len(set(values))==1 for values in physical.values())
    assert all(all(p==prefix[0] for p in prefix) for prefix in prefixes.values())
    totals={a:{'attempted':sum(r['arm']==a for r in rows),
        'clean':sum(r['arm']==a and r['status'] not in BAD for r in rows),
        'successes':sum(r['arm']==a and r['success'] for r in rows),
        **{k:sum(r[k] for r in rows if r['arm']==a) for k in ('model_calls','environment_steps','cost_usd','invalid_actions','compiled_search_actions','native_multi_calls','actual_tail_interruptions','discarded_primitives','terminal_queue_cleanups')}} for a in cfg['arms']}
    comparisons={}
    for baseline in ('react','skillopt','matched_single','matched_fixed'):
        repeats=[]
        for rep in (1,2):
            pairs=[]
            for t in cfg['tasks']:
                if t['replicate']!=rep:continue
                a=next(r for r in rows if r['task_id']==t['id'] and r['arm']=='matched_dynamic')
                b=next(r for r in rows if r['task_id']==t['id'] and r['arm']==baseline)
                if a['status'] in BAD or b['status'] in BAD:continue
                pairs.append((a,b))
            wins=sum(a['success'] and not b['success'] for a,b in pairs)
            losses=sum(b['success'] and not a['success'] for a,b in pairs)
            calls=sum(a['model_calls']-b['model_calls'] for a,b in pairs)
            dollars=sum(a['cost_usd']-b['cost_usd'] for a,b in pairs)
            repeats.append({'replicate':rep,'clean_pairs':len(pairs),'wins':wins,'losses':losses,'model_call_delta':calls,'dollar_delta':dollars,
                'gain':len(pairs)>=2 and ((wins>0 and losses==0) or (wins==losses==0 and calls<0 and dollars<=0))})
        comparisons[baseline]={'repeats':repeats,'both_repeats_gain':all(r['gain'] for r in repeats)}
    record={'checks_passed':True,'complete':True,'totals':totals,'dynamic_vs':comparisons,'invalid_jobs':invalid,
        'identical_initial_public_states':True,'identical_pre_model_operator_prefixes':True,
        'equal_macros_disabled':True,'shared_operator_state_provided_all_arms':True,
        'native_queue_increment_supported':comparisons['matched_single']['both_repeats_gain'] and totals['matched_dynamic']['native_multi_calls']>0,
        'interrupt_increment_supported':comparisons['matched_fixed']['both_repeats_gain'] and totals['matched_dynamic']['actual_tail_interruptions']>0,
        'batch':batch_accounting(root),'rows':rows,
        'config_sha256':digest(root/'config.json'),'summary_sha256':digest(root/'summary.json'),'audit_source_sha256':digest(Path(__file__)),
        'limitations':['Attribution on previously exposed four physical tasks with two repeats, not independent benchmark generalization.',
         'Released GPT5.5 SkillOpt checkpoint transferred to Qwen, no fresh SkillOpt optimization.',
         'Original inference templates preserved with common operator summary/resource/brief-output additions and strict no-retry parsing; matched ablations isolate execution controller.']}
    save(root/'validation_audit.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='rows'},ensure_ascii=False))
    return record

if __name__=='__main__':audit()

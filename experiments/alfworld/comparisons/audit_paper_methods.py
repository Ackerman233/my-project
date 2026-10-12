"""Audit frozen paper controls, matched resets, provenance and actual gains."""
from pathlib import Path
import argparse
from experiments.interrupt_recovery_ablation import read,digest,config_hash
from experiments.bounded_research95 import check_frozen,batch_accounting
from experiments.paper_method_trial import BAD
from experiments.puzzle_pilot import save


def audit(root):
    root=Path(root);cfg=read(root/'config.json');check_frozen(root,cfg)
    s=read(root/'summary.json');assert s['complete']
    gains={};invalid=[]
    for t in cfg['tasks']:
        initial=[]
        for arm in cfg['arms']:
            out=root/'online'/t['id']/arm;r=read(out/'result.json')
            trace=read(out/'conversation.json') if (out/'conversation.json').exists() else []
            calls=read(out/'model_calls.json');initial.append(r['initial_sha256'])
            assert r['initial_sha256']==config_hash(read(out/'initial.json'))
            assert r['model_calls']==len(calls)<=cfg['settings']['max_calls']
            assert r['environment_steps']==len(trace)<=cfg['settings']['max_steps']
            assert r['compiled_search_actions']==sum(x['action_origin']=='compiled_public_search' for x in trace)
            for x in trace:
                if x['action_origin']=='compiled_public_search':
                    assert x['action'] in x['admissible_before'] and x['operator']['actions']==[x['action']]
            if r['status'] in BAD:invalid.append({'task':t['id'],'arm':arm,'status':r['status']})
        assert len(set(initial))==1
    if cfg['stage']!='dev':
        for a in cfg['arms']:
            if a=='rule':continue
            pairs=[p for p in s['pairs'] if p['arms']['rule']['status'] not in BAD and p['arms'][a]['status'] not in BAD]
            wins=sum(p['arms'][a]['success'] and not p['arms']['rule']['success'] for p in pairs)
            losses=sum(p['arms']['rule']['success'] and not p['arms'][a]['success'] for p in pairs)
            calls=sum(p['arms'][a]['model_calls']-p['arms']['rule']['model_calls'] for p in pairs)
            cost=sum(p['arms'][a]['cost_usd']-p['arms']['rule']['cost_usd'] for p in pairs)
            gains[a]={'clean_pairs':len(pairs),'wins':wins,'losses':losses,'model_call_delta':calls,'dollar_delta':cost,
                'task_gain_demonstrated':len(pairs)>=2 and ((wins>0 and losses==0) or (wins==losses==0 and calls<0 and cost<=0))}
            if a=='value':
                gains[a]['actual_learned_interrupts']=s['arms'][a]['learned_interrupts']
                gains[a]['mechanism_gain_supported']=gains[a]['task_gain_demonstrated'] and s['arms'][a]['learned_interrupts']>0
        history=read(root/'history_exclusion.json');assert not set(history['selected_scene_names'])&set(history['used_scene_names'])
        for p,sha in history['historical_files'].items():assert digest(Path(p))==sha
    record={'protocol_checks_passed':True,'gain_vs_rule':gains,'invalid_jobs':invalid,'batch':batch_accounting(root),
        'config_sha256':digest(root/'config.json'),'summary_sha256':digest(root/'summary.json'),
        'limitations':['Small sample in two ALFWorld families, no cross-environment evidence.',
            'Workflow operator hand-designed with exact public admissible commands. Its call savings include program search and pickup, not learned interruption or native model batching.',
            'Termination trained on noisy single historical branch realizations, not exact causal expected values.']}
    save(root/'validation_audit.json',record);print(record)
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');audit(p.parse_args().root)

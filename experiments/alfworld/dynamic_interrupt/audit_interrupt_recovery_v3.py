"""Offline replay of REAL saved actions, plus clearly labeled SYNTHETIC example.

No API client. No paid model calls. Recorded replay is not a counterfactual success.
"""
import hashlib,json
from pathlib import Path
from dynamic_skill.control_only_evolution import ControlExecutor,ControlPolicy
from experiments.puzzle_pilot import save

ROOT=Path('results/interrupt_recovery_v3')
def read(p):return json.loads(p.read_text())
def run():
    source=Path('results/micro_evolution_qwen36_20260929/control_parameter_trial/dev')
    reports=[]
    for p in sorted(source.glob('*/predictions/*/result.json')):
        row=read(p);out=p.parent;initial=read(out/'initial.json');calls=read(out/'model_calls.json');trace=read(out/'conversation.json')
        c=ControlExecutor(ControlPolicy(**row['control_policy']),state_check=True)
        c.start(initial['observation'],initial['task']);checks=[];divergence=None
        for i,step in enumerate(trace):
            if step['model_called']:
                if c.pending_actions:divergence={'step':i,'reason':'recorded replanning with pending actions'};break
                c.load_chunk(calls[step['model_call_id']-1]['actions'])
            if not c.pending_actions:divergence={'step':i,'reason':'new pause; cannot claim recorded suffix as new execution'};break
            action=c.next_action();assert action==step['action']
            c.before_action(action,trace[i-1]['env_feedback'] if i else initial['observation'],step['admissible_before'])
            event=c.observe(action,step['env_feedback'],reward=step['reward'],done=step['done'],info={'won':step['won']})
            checks.append(c.last_execution_audit)
            if event.decision.kind.value!=step['controller_decision']:
                divergence={'step':i,'reason':'new decision differs','original':step['controller_decision'],'new':event.decision.kind.value};break
        reports.append({'source':str(out),'source_trace_sha256':hashlib.sha256((out/'conversation.json').read_bytes()).hexdigest(),
                        'kind':'real_recorded_trace_replay','steps_replayed':len(checks),'divergence':divergence,'checks':checks})
    before=ControlExecutor(ControlPolicy(),state_check=False);after=ControlExecutor(ControlPolicy(),state_check=True)
    example={}
    for name,c in [('before',before),('after',after)]:
        c.start('On the desk 1 you see bowl 1.','put a bowl on countertop')
        c.load_chunk(['take bowl 1 from desk 1','move bowl 1 to countertop 1'])
        action=c.next_action();c.before_action(action,'On the desk 1 you see bowl 1.',[action])
        e=c.observe(action,'You pick up the mug 1 from the desk 1.')
        example[name]={'decision':e.decision.kind.value,**c.last_execution_audit}
    summary={'paid_calls':0,'real_trace_count':len(reports),'real_recorded_actions':sum(r['steps_replayed'] for r in reports),
             'real_divergences':[r['divergence'] for r in reports if r['divergence']],
             'synthetic_counterexample':example,'note':'Synthetic object mismatch is injected; not claimed to occur naturally in ALFWorld. Replay is code verification, not new model performance.'}
    save(ROOT/'offline_replay.json',reports);save(ROOT/'offline_audit.json',summary);print(json.dumps(summary))

def live_environment_example():
    """Scripted plans against real env; inventory is executed and counted, not queried secretly."""
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    from dynamic_skill.control_only_evolution import run_control_episode
    from experiments.puzzle_pilot import Settings
    src=Path('results/micro_evolution_qwen36_20260929/control_parameter_trial/config.json')
    task=read(src)['dev'][0];rows=[]
    for enabled in (False,True):
        class ScriptedPlanner:
            def __init__(self):
                self.plans=iter([['go to desk 1'],['take bowl 1 from desk 1','move bowl 1 to desk 1',
                                                    'inventory','move bowl 1 to desk 1'],['take bowl 1 from desk 1']])
            def before_call(self):pass
            def __call__(self,prompt):
                return {'response':'<thought>Scripted offline mechanism test.</thought><actions>'+json.dumps(next(self.plans))+'</actions>',
                        'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}
        env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,is_train=False,specific_gamefiles=[task['gamefile']])
        out=ROOT/'scripted_live_environment_with_replan'/('state_on' if enabled else 'state_off')
        try:
            row=run_control_episode(env,task,Settings(model='OFFLINE_SCRIPT_NOT_MODEL',max_calls=3,max_steps=7),
                out,ScriptedPlanner(),'Offline evidence check only.',ControlPolicy(),state_check=enabled,history_evidence=True)
        finally:env.close()
        p=out/'predictions'/row['id'];trace=read(p/'conversation.json')
        inv=next(t for t in trace if t['action']=='inventory')
        assert inv['env_feedback']=='You are not carrying anything.'
        assert inv['state_check']['status']=='contradicted'
        assert inv['semantic_pause']==enabled
        assert len(trace)==(5 if enabled else 6)
        assert inv['actual_discarded_actions']==(['move bowl 1 to desk 1'] if enabled else [])
        calls=read(p/'model_calls.json')
        assert calls[2]['historical_evidence_injected']
        if enabled:
            assert calls[2]['state_recovery_received']['last_check']['status']=='contradicted'
            assert trace[-1]['original_failure_verified_resolved']
        row.update(evidence_kind='scripted_planner_real_environment_NOT_model_performance',paid_calls=0)
        save(p/'result.json',row)
        rows.append({'state_check':enabled,'environment_steps':len(trace),'inventory_action_count':1,
                     'paid_calls':0,'scripted_planner_calls':3,'inventory_step':inv,
                     'next_planning_context':calls[2],'result_path':str(p)})
    save(ROOT/'live_environment_audit.json',{'kind':'scripted real environment, not model evaluation','rows':rows})
    print(json.dumps({'live_environment_verified':True,'paid_calls':0,'steps_off':6,'steps_on':5,'inventory_count_each':1}))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--live-environment',action='store_true')
    parser.add_argument('--out-root',type=Path,default=ROOT);args=parser.parse_args()
    ROOT=args.out_root
    if args.live_environment:live_environment_example()
    else:run()

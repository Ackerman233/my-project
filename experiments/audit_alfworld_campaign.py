"""No API access: replay recorded ALFWorld commands and validate accounting."""
import json
from pathlib import Path
from experiments.alfworld_campaign import OUT, MODELS, MODES, prompt_for, SYSTEM
from experiments.puzzle_pilot import save
from dynamic_skill.skill_executor import ChunkExecutor


def run():
    from skillopt.envs.alfworld.rollout import build_alfworld_env
    config=json.loads((OUT/'config.json').read_text())
    results=[json.loads(s) for s in (OUT/'results.jsonl').read_text().splitlines()]
    assert len(results)==len({r['id'] for r in results})
    prompt_checks=0; step_checks=0
    for task in config['tasks']:
        env=build_alfworld_env(env_num=1,eval_dataset='eval_in_distribution',seed=42,
                             is_train=False,specific_gamefiles=[task['gamefile']])
        try:
            for row in [r for r in results if r['task_id']==task['id']]:
                directory=OUT/'predictions'/row['id']
                calls=json.loads((directory/'model_calls.json').read_text()) if (directory/'model_calls.json').exists() else []
                trace=json.loads((directory/'conversation.json').read_text()) if (directory/'conversation.json').exists() else []
                initial=json.loads((directory/'initial.json').read_text())
                obs,info=env.reset({})
                assert obs['anchor'][0]==initial['observation']
                assert env.tasks[0]==initial['task']
                k=1 if row['mode']=='react' else 4
                ctl=ChunkExecutor(chunk_size=k,dynamic=row['mode']=='dynamic_interrupt')
                ctl.start(obs['anchor'][0],task=env.tasks[0])
                for i,step in enumerate(trace):
                    call=calls[step['model_call_id']-1]
                    if step['model_called']:
                        assert ctl.needs_plan
                        prompt=prompt_for(env.tasks[0],obs['anchor'][0],list(env.envs.get_admissible_commands[0]),
                            trace[:i],config['skill'],min(k,50-i),ctl.last_replan_reason)
                        assert prompt==call['prompt'] and call['system_prompt']==SYSTEM
                        prompt_checks+=1
                        ctl.load_chunk(call['actions'])
                    action=ctl.next_action()
                    assert action==step['action']
                    admissible=list(env.envs.get_admissible_commands[0])
                    assert admissible==step['admissible_before']
                    assert (action.lower() in {a.lower() for a in admissible})==step['valid']
                    obs,reward,done,info=env.step(['<think></think><action>'+action+'</action>'])
                    assert obs['anchor'][0]==step['env_feedback']
                    assert float(reward[0])==step['reward'] and bool(done[0])==step['done']
                    event=ctl.observe(action,obs['anchor'][0],reward=float(reward[0]),done=bool(done[0]),info=info[0])
                    assert event.decision.kind.value==step['controller_decision']
                    assert list(event.discarded_actions)==step['discarded_actions']
                    assert list(ctl.pending_actions)==step['remaining_actions']
                    step_checks+=1
                assert len(trace)==row['environment_steps'] and len(calls)==row['model_calls']
                assert sum(not s['valid'] for s in trace)==row['invalid_actions']
                assert row['success']==bool(trace and trace[-1]['done'] and (trace[-1]['won'] or trace[-1]['reward']>0))
                for field in ('prompt_tokens','completion_tokens','total_tokens'):
                    assert sum((c.get('usage') or {}).get(field,0) or 0 for c in calls)==row[field]
                for c in calls:
                    assert c['status']!='pending'
                    details=(c.get('usage') or {}).get('completion_tokens_details') or {}
                    assert not c.get('reasoning_content') and not (details.get('reasoning_tokens') or 0)
        finally: env.close()
    pairs=0
    for task in config['tasks']:
        for mode in MODES:
            starts=[]
            for model in MODELS:
                p=OUT/'predictions'/(model+'__'+task['id']+'__'+mode)/'model_calls.json'
                if p.exists():
                    calls=json.loads(p.read_text())
                    if calls: starts.append(calls[0]['prompt'])
            if len(starts)==2:
                assert starts[0]==starts[1]; pairs+=1
    audit={'episodes':len(results),'environment_steps_replayed':step_checks,
        'executed_call_prompts_verified':prompt_checks,'paired_initial_prompts_verified':pairs,
        'queue_and_interrupt_audit':True,'environment_replay':True,'usage_sums':True}
    save(OUT/'audit.json',audit); print(json.dumps(audit),flush=True)


if __name__=='__main__': run()

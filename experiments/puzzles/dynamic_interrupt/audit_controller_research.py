"""Replay actual recorded environment actions; audit, never new model success."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from experiments.controller_research import ROOT, LEDGER, AUTH, check_config, sha
from dynamic_skill.puzzles import make_environment
from dynamic_skill.research_controller import Controller, POLICIES
from experiments.puzzle_pilot import save


def main():
    cfg=json.loads((ROOT/'config.json').read_text()); check_config(cfg)
    ledger=json.loads(LEDGER.read_text()); episodes=0; requests=0; native=0; multi=0; missing_reasoning_usage=0
    for path in ROOT.glob('*/*/*/result.json'):
        row=json.loads(path.read_text()); out=path.parent
        task=next(t for t in cfg['splits'][row['split']] if t['id']==row['task_id'])
        env=make_environment(task); c=Controller(POLICIES[row['policy']])
        calls=json.loads((out/'calls.json').read_text()) if (out/'calls.json').exists() else []
        trajectory=json.loads((out/'trajectory.json').read_text()) if (out/'trajectory.json').exists() else []
        paid=[r for r in ledger['requests'] if r['label']==str(out)]
        assert len(paid)==len(calls)==row['paid_requests']==row['model_calls']
        for call in calls:
            if cfg['settings']['model']=='gpt-5.4':
                assert call['reasoning_effort']=='none' and call['max_completion_tokens']==256
            else:
                assert call['enable_thinking'] is False and call['max_tokens']==256
            assert call['system_prompt']==cfg['system']
            details=(call.get('usage') or {}).get('completion_tokens_details') or {}
            missing_reasoning_usage+= details.get('reasoning_tokens') is None
            n=details.get('reasoning_tokens') or 0
            native+=n; assert n==0
            assert not call.get('reasoning_content')
            multi+=len(call.get('actions',[]))>1
        source={i+1:iter(call.get('actions',[])) for i,call in enumerate(calls)}
        for step in trajectory:
            assert env.render()==step['board_before']
            assert next(source[step['call_id']])==step['action']
            result=env.step(step['action'])
            assert result.observation==step['board_after'] and result.valid==step['valid']
            event=c.observe(step['board_before'],step['action'],result.valid,result.observation,step['tail_before_decision'])
            for key,value in event.items(): assert step[key]==value
        assert row['success']==bool(trajectory and trajectory[-1]['won'])
        episodes+=1; requests+=len(paid)
    out={'pass':True,'episodes':episodes,'requests':requests,'native_reasoning_tokens':native,
         'multi_action_calls':multi,'calls_without_reasoning_usage_field':missing_reasoning_usage,'config_sha':cfg['hash'],
         'note':'Replay checks recorded real trajectories; not counted as extra model evaluations.'}
    save(ROOT/'audit.json',out); print(json.dumps(out))

if __name__=='__main__': main()

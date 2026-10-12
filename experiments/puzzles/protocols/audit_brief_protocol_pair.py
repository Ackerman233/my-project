"""Read-only model trajectory and budget verification, plus protocol adherence."""
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.campaign_budget import CampaignBudget
from dynamic_skill.puzzles import make_environment
from experiments.puzzle_pilot import save
from experiments.cold_start_gate import gate
from experiments.brief_protocol_pair import ROOT, AUTH


def main():
    read=lambda p:json.loads(p.read_text())
    cfg=read(ROOT/'config.json'); rows=read(ROOT/'results.json'); budget=read(ROOT/'budget_snapshot.json')
    requests=budget['requests'][budget[AUTH]['previous_requests']:]
    settings=[dict(s) for s in cfg['settings'].values()]
    for s in settings:
        s.pop('action_only'); s.pop('output_protocol')
    assert settings[0]==settings[1]
    ids=[]; totals={}
    for arm,results in rows.items():
        t={'calls':0,'reasoning_tokens':0,'protocol_violations':0,'truncated_responses':0,
           'latency_seconds':0,'estimated_cost_usd':0,'total_tokens':0,'max_reason_words':0}
        for row in results:
            task=next(x for x in cfg['tasks'] if x['id']==row['task_id'])
            env=make_environment(task)
            folder=ROOT/arm/'predictions'/row['id']
            calls=read(folder/'model_calls.json')
            trace=read(folder/'conversation.json') if (folder/'conversation.json').exists() else []
            text=cfg['base_skills'][row['benchmark']]
            skill_hash=hashlib.sha256(text.encode()).hexdigest()
            assert skill_hash==row['base_skill_sha256']==cfg['base_hashes'][row['benchmark']]
            for c in calls:
                assert c['reasoning_effort']=='none' and c['base_skill_sha256']==skill_hash
                assert text.split('\n## Output\n')[0] in c['prompt']
                if c.get('response_id'): ids.append(c['response_id'])
                t['calls']+=1; t['latency_seconds']+=c['latency_seconds']
                usage=c.get('usage') or {}
                t['total_tokens']+=usage.get('total_tokens',0)
                t['reasoning_tokens']+=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
                assert not c.get('reasoning_content')
                t['truncated_responses']+=c.get('finish_reason')=='length'
                if c['status']=='ok':
                    assert len(c['actions'])==1
                    pattern=(r'\s*<actions>.*?</actions>\s*' if arm=='action_only' else
                             r'\s*<reason>([^<>]+)</reason>\s*<actions>.*?</actions>\s*')
                    match=re.fullmatch(pattern,c['response'],re.S)
                    violation=match is None
                    if match and arm=='brief_reason':
                        words=len(match.group(1).split()); t['max_reason_words']=max(words,t['max_reason_words'])
                        violation=words>30
                    t['protocol_violations']+=violation
            for step in trace:
                c=calls[step['model_call_id']-1]
                assert step['action']==c['actions'][0]
                replay=env.step(step['action'])
                assert replay.valid==step['valid'] and replay.observation==step['board_after'] and replay.won==step['won']
            assert env.solved==row['success']
            t['estimated_cost_usd']+=row['estimated_cost_usd'] or 0
        t['gate']=gate(results,cfg['tasks'])
        totals[arm]=t
    assert set(ids)=={r['response_id'] for r in requests if r.get('response_id')}
    assert sum(t['calls'] for t in totals.values())==len(requests)
    assert all(t['reasoning_tokens']==0 for t in totals.values())
    paired=[]
    for task in cfg['tasks']:
        p={'task_id':task['id']}
        for arm,results in rows.items():
            r=next((r for r in results if r['task_id']==task['id']),None)
            p[arm]=None if r is None else {k:r[k] for k in ('success','status','model_calls','invalid_actions')}
        paired.append(p)
    out={'verified':True,'arms':totals,'paired':paired,'paid_requests':len(requests),
         'incremental_conservative_usd':sum(r['budget_charge_usd'] for r in requests),
         'cumulative_guarded_usd':CampaignBudget.effective_spend(budget),'limit_usd':budget['limit_usd'],
         'note':'One sample per task/arm, no statistical generalization claim; replay is audit, not new performance.'}
    save(ROOT/'audit.json',out); print(json.dumps(out,indent=2))

if __name__=='__main__': main()

"""Read-only environment replay; no API calls or modifications to shared ledger."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.campaign_budget import CampaignBudget
from dynamic_skill.puzzles import make_environment
from experiments.puzzle_pilot import save
from experiments.cold_start_gate import gate


def main():
    root=Path('results/cold_start_gate_20260928')
    read=lambda p:json.loads(p.read_text())
    config=read(root/'config.json');rows=read(root/'results.json')
    tasks={t['id']:t for t in config['tasks']}
    calls=0;reasoning=0
    for row in rows:
        env=make_environment(tasks[row['task_id']])
        folder=root/'predictions'/row['id']
        records=read(folder/'model_calls.json')
        trace=read(folder/'conversation.json') if (folder/'conversation.json').exists() else []
        text=config['base_skills'][row['benchmark']]
        expected=hashlib.sha256(text.encode()).hexdigest()
        assert expected==row['base_skill_sha256']
        for record in records:
            assert text in record['prompt']
            assert record['base_skill_sha256']==expected
            if record['status']=='ok':assert len(record['actions'])==1
            reasoning+=((record.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
        for step in trace:
            record=records[step['model_call_id']-1]
            assert step['action']==record['actions'][0]
            result=env.step(step['action'])
            assert result.valid==step['valid'] and result.observation==step['board_after']
        assert env.solved==row['success']
        calls+=len(records)
    assert gate(rows,config['tasks'])==read(root/'gate.json')
    budget=read(root/'budget_snapshot.json')
    new=budget['requests'][budget['cold_start_authorization']['previous_requests']:]
    actual_ids={r['response_id'] for r in new if r.get('response_id')}
    observed_ids=set(); local_halts=[]
    for path in root.glob('predictions/*/model_calls.json'):
        for record in read(path):
            if record.get('response_id'): observed_ids.add(record['response_id'])
            if record.get('error_type')=='CampaignHalt':
                # Verify this is an unsent budget stop, not an API failure.
                assert not record.get('response_id')
                upper=((len(record['prompt'].encode())+2048)*budget['input_usd_per_million']+128*budget['output_usd_per_million'])/1e6
                assert CampaignBudget.effective_spend(budget)+upper>budget['limit_usd']
                local_halts.append({'episode':path.parent.name,'call_id':record['call_id'],
                    'correct_classification':'unsent_budget_stop','required_reservation_usd':upper})
    assert observed_ids==actual_ids and calls==len(new)+len(local_halts)
    result={'verified':True,'episodes':len(rows),'attempt_records':calls,'paid_requests':len(new),
        'unsent_guard_stops':local_halts,'reported_reasoning_tokens':reasoning,
        'incremental_estimate_usd':sum(r['budget_charge_usd'] for r in new),
        'effective_budget_at_finish_usd':CampaignBudget.effective_spend(budget),
        'gate':gate(rows,config['tasks']),
        'rows':[{k:r[k] for k in ('task_id','success','status','model_calls','environment_steps','invalid_actions')} for r in rows]}
    save(root/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

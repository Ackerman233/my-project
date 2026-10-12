"""Replay only the paid model's recorded actions; never count replay as new performance."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
from dynamic_skill.campaign_budget import CampaignBudget
from dynamic_skill.puzzles import make_environment
from experiments.cold_start_gate import gate
from experiments.paper_cold_start_gate import ROOT
from experiments.puzzle_pilot import save


def main():
    read = lambda p: json.loads(p.read_text())
    config, rows = read(ROOT/'config.json'), read(ROOT/'results.json')
    budget = read(ROOT/'budget_snapshot.json')
    requests = budget['requests'][budget['paper_cold_start_authorization']['previous_requests']:]
    ids, calls, reasoning, tokens, latency = [], 0, 0, 0, 0
    for row in rows:
        task = next(t for t in config['tasks'] if t['id'] == row['task_id'])
        env = make_environment(task)
        folder = ROOT/'predictions'/row['id']
        records = read(folder/'model_calls.json')
        trace = read(folder/'conversation.json') if (folder/'conversation.json').exists() else []
        text = config['base_skills'][row['benchmark']]
        skill_hash = hashlib.sha256(text.encode()).hexdigest()
        assert skill_hash == row['base_skill_sha256'] == config['base_hashes'][row['benchmark']]
        for record in records:
            assert text in record['prompt'] and '## Current Observation' in record['prompt']
            assert record['reasoning_effort'] == 'none'
            assert record['base_skill_sha256'] == skill_hash
            assert record['status'] == 'ok' and len(record['actions']) == 1
            assert record.get('usage') and not record.get('reasoning_content')
            ids.append(record['response_id'])
            reasoning += (record['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
            tokens += record['usage']['total_tokens']
            latency += record['latency_seconds']
        for step in trace:
            record = records[step['model_call_id']-1]
            assert step['action'] == record['actions'][0]
            assert '## Remaining Steps\n' + str(6-step['step']) in record['prompt']
            result = env.step(step['action'])
            assert result.valid == step['valid'] and result.observation == step['board_after']
            assert result.won == step['won']
        assert env.solved == row['success']
        calls += len(records)
    assert calls == len(requests) and set(ids) == {r['response_id'] for r in requests}
    assert len(ids) == len(set(ids)) and reasoning == 0
    assert gate(rows, config['tasks']) == read(ROOT/'gate.json')
    out = {'verified': True, 'paid_requests': calls, 'reported_reasoning_tokens': reasoning,
           'total_tokens': tokens, 'api_latency_seconds': latency,
           'incremental_conservative_estimate_usd': sum(r['budget_charge_usd'] for r in requests),
           'cumulative_guarded_spend_usd': CampaignBudget.effective_spend(budget),
           'limit_usd': budget['limit_usd'], 'gate': gate(rows, config['tasks']),
           'note': 'Replay validates original model traces; it is not an additional model run or provider invoice.'}
    save(ROOT/'audit.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__': main()

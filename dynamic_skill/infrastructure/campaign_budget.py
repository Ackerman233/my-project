"""Durable conservative budget accounting; caller holds a process-level lock."""
import json
from pathlib import Path
import time


class CampaignHalt(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


class CampaignBudget:
    def __init__(self, path, *, deadline, limit=5.0, clock=time.time,
                 input_rate=1.0, output_rate=5.0, max_requests=600, max_unknown=4):
        self.path, self.clock = Path(path), clock
        if not self.path.exists():
            self.write({'limit_usd': limit, 'deadline_unix': deadline, 'reservation_usd': 0.5,
                        'input_usd_per_million': input_rate, 'output_usd_per_million': output_rate,
                        'max_requests': max_requests, 'max_unknown_requests': max_unknown,
                        'consecutive_errors': 0, 'cooldown_until': 0, 'policy_violation': False,
                        'requests': [], 'note': 'Conservative client ledger, not verified provider billing. Unknown usage retains reservation.'})

    def read(self):
        return json.loads(self.path.read_text())

    def write(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(self.path)

    def reserve(self, label, reservation_usd=None):
        data = self.read()
        now = self.clock()
        if now >= data['deadline_unix']:
            raise CampaignHalt('deadline_stop')
        if data['policy_violation']:
            raise CampaignHalt('thinking_policy_stop')
        if sum(r['status'] in {'pending', 'unknown_usage'} for r in data['requests']) >= data['max_unknown_requests']:
            raise CampaignHalt('unknown_cost_stop')
        if now < data['cooldown_until']:
            raise CampaignHalt('api_cooldown')
        reserve = data['reservation_usd'] if reservation_usd is None else reservation_usd
        import math
        if not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve <= 0:
            raise ValueError('reservation must be finite and positive')
        spent = self.effective_spend(data)
        if spent + reserve > data['limit_usd'] + 1e-9:
            raise CampaignHalt('budget_stop')
        if len(data['requests']) >= data['max_requests']:
            raise CampaignHalt('request_cap_stop')
        index = len(data['requests'])
        data['requests'].append({'id': index + 1, 'label': label, 'started_unix': now,
                                 'status': 'pending', 'budget_charge_usd': reserve,
                                 'initial_reservation_usd': reserve})
        self.write(data)
        return index

    @staticmethod
    def effective_spend(data):
        checkpoint = data.get('billing_checkpoint')
        if not checkpoint:
            return sum(r['budget_charge_usd'] for r in data['requests'])
        return (checkpoint['user_reported_total_usd'] + checkpoint['unsettled_reserve_usd']
                + sum(r['budget_charge_usd'] for r in data['requests'][checkpoint['request_count']:]))

    def settle(self, index, response=None, error=None):
        data = self.read()
        row = data['requests'][index]
        assert row['status'] == 'pending', 'reservation must settle exactly once'
        response = response or {}
        usage = response.get('usage')
        known = isinstance(usage, dict) and all(isinstance(usage.get(k), int) and usage[k] >= 0
                                                for k in ('prompt_tokens', 'completion_tokens'))
        if known:
            row.update(status='returned_usage', usage=usage,
                       budget_charge_usd=(usage['prompt_tokens'] * data['input_usd_per_million']
                                          + usage['completion_tokens'] * data['output_usd_per_million']) / 1e6)
        else:
            row['status'] = 'unknown_usage'  # Keep the entire reservation, including errors.
        row['finished_unix'] = self.clock()
        row['response_id'] = response.get('response_id')
        row['error_type'] = type(error).__name__ if error else None
        details = (usage or {}).get('completion_tokens_details') or {}
        if response.get('reasoning_content') or (details.get('reasoning_tokens') or 0) > 0:
            data['policy_violation'] = True
        if error or not known:
            data['consecutive_errors'] += 1
            if data['consecutive_errors'] >= 2:
                data['cooldown_until'] = self.clock() + 900
        else:
            data['consecutive_errors'] = 0
        self.write(data)


class BudgetPlanner:
    def __init__(self, planner, budget, label):
        self.planner, self.budget, self.label = planner, budget, label
        self.reservation = None

    def before_call(self, reservation_usd=None):
        self.reservation = self.budget.reserve(self.label, reservation_usd=reservation_usd)

    def __call__(self, prompt):
        if self.reservation is None:
            raise RuntimeError('budget must be reserved before any paid call')
        index, self.reservation = self.reservation, None
        try:
            response = self.planner(prompt)
        except Exception as exc:
            self.budget.settle(index, error=exc)
            raise
        self.budget.settle(index, response=response)
        return response

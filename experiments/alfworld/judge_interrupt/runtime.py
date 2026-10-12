"""Shared actor, primitive operators and TOTAL request allowance for all arms."""
from dataclasses import asdict, replace
import json

from dynamic_skill.controllers.control_only_evolution import ControlExecutor
from dynamic_skill.controllers.judge import JudgeConfig, JudgeExecutor, LocalJudge, HardEvidenceExecutor
from dynamic_skill.infrastructure.campaign_budget import CampaignHalt
from dynamic_skill.alfworld.micro_loop_protocol import SYSTEM, make_prompt, parse_response
from dynamic_skill.alfworld.alfworld_recovery import recovery_prompt
from experiments.alfworld.common.alfworld_campaign import run_episode
from experiments.common.puzzle_pilot import save


class EpisodeAllowance:
    def __init__(self, transport, label, maximum, judge_transport=None):
        self.transport, self.label, self.maximum = transport, label, maximum
        self.judge_transport = judge_transport or transport
        self.actor_calls = self.judge_calls = 0

    @property
    def remaining(self):
        return self.maximum - self.actor_calls - self.judge_calls

    def before_call(self):
        if self.remaining <= 0:
            raise CampaignHalt('model_call_limit')

    def __call__(self, prompt):
        self.before_call()
        self.actor_calls += 1
        return self.transport.request(prompt, system=SYSTEM,
            max_tokens=self.transport.settings.max_tokens, label=self.label + '/actor')

    def judge_request(self, prompt, **kwargs):
        self.before_call()
        self.judge_calls += 1
        return self.judge_transport.request(prompt, label=self.label + '/judge', **kwargs)


class SharedBudgetJudge(LocalJudge):
    def __init__(self, allowance, config, sink):
        super().__init__(allowance.judge_request, config, sink)
        self.allowance = allowance

    def decide(self, context):
        # Keep one model call available to act on the judgment; checking with no
        # possible recovery would pay for an unusable decision.
        if self.allowance.remaining < 2:
            return {'decision': 'defer', 'status': 'shared_call_budget'}
        return super().decide(context)


def evaluate(env, task, genome, transport, directory, arm, judge_config, judge_transport=None):
    settings, policy = transport.settings, genome.policy
    config = JudgeConfig(**judge_config)
    allowance = EpisodeAllowance(transport, str(directory / task['id']), settings.max_calls, judge_transport)
    judge = None
    if arm == 'rules':
        ctl = HardEvidenceExecutor(policy.executable(), dynamic=True,
            authoritative_admissibility=config.authoritative_admissibility,
            state_check=policy.state_check, discovery_interrupt=policy.discovery_interrupt)
    else:
        judge = SharedBudgetJudge(allowance, JudgeConfig(**{**judge_config, 'role': arm}),
            lambda rows: save(directory / 'judge_calls.json', rows))
        ctl = JudgeExecutor(policy.executable(), dynamic=True, judge=judge,
            authoritative_admissibility=config.authoritative_admissibility,
            state_check=policy.state_check, discovery_interrupt=policy.discovery_interrupt)

    # Match the incumbent actor prompt, including public acquisition history.
    # All experimental arms share this function and the exact same frozen skill.
    def prompt(ignored, **context):
        limit = min(ctl.planning_limit, settings.max_steps - len(context['trace']))
        body = make_prompt(genome.skill_text, limit, 'Replan from the current observation.')(ignored, **context)
        enhanced = bool(policy.history_evidence and context['anomaly'])
        if enhanced:
            body = recovery_prompt(body, **context, output_protocol='thought_actions')
        recovery_info = None
        if policy.state_check and context['anomaly']:
            recovery_info = {'last_check': json.loads(json.dumps(ctl.last_execution_audit.get('state_check'))),
                'failures': json.loads(json.dumps(ctl.failures)),
                'planning_at_step': len(context['trace']),
                'note': 'A recovery counter reaching zero is not evidence the failed condition is resolved. '
                        'If possession is unknown, inventory is a real environment action to verify it.'}
            for evidence in ([recovery_info['last_check'].get('evidence')] if recovery_info['last_check'] else []) + [
                    f.get('resolution_evidence') for f in recovery_info['failures']]:
                if evidence:
                    evidence.update(age_steps=len(context['trace'])-1-evidence['observed_at_step'],
                                    historical=True, current_state_guaranteed=False)
            body += '\nAcquisition recovery context:\n' + json.dumps(recovery_info)
        ctl.planning_context = {'effective_action_limit': limit,
            'generic_anomaly_prompt_used': bool(context['anomaly']),
            'historical_evidence_injected': enhanced, 'recovery_prompt_used': enhanced,
            'single_step_recovery_active': ctl.recovery_remaining > 0,
            'recovery_remaining': ctl.recovery_remaining, 'anomaly_received': context['anomaly'],
            'state_recovery_received': recovery_info, 'failure_resolution': json.loads(json.dumps(ctl.failures))}
        return body

    row = run_episode(env, task, 'dynamic_interrupt', replace(settings, chunk_size=policy.max_actions),
        directory, allowance, genome.skill_text, prompt_transform=prompt,
        response_parser=lambda text, limit: parse_response(text, min(limit, ctl.planning_limit)),
        system_prompt=SYSTEM, executor=ctl)
    folder = directory / 'predictions' / row['id']
    trace = json.loads((folder / 'conversation.json').read_text()) if (folder / 'conversation.json').exists() else []
    requests = [r for r in transport.budget.read()['requests'] if r['label'].startswith(allowance.label + '/')]
    records = judge.records if judge else []
    row.update(arm=arm, genome_hash=genome.hash, joint_policy=asdict(policy),
        authoritative_admissibility=config.authoritative_admissibility,
        hard_head_guard_events=sum(r.get('hard_head_guard', False) for r in trace),
        judge_model=allowance.judge_transport.settings.model if judge else None,
        actor_calls=row['model_calls'], judge_calls=len(records),
        total_model_calls=row['model_calls'] + len(records),
        model_calls=row['model_calls'] + len(records),  # gates must count judge calls
        cost_usd=sum(r['budget_charge_usd'] for r in requests),
        actor_cost_usd=sum(r['budget_charge_usd'] for r in requests if r['label'].endswith('/actor')),
        judge_cost_usd=sum(r['budget_charge_usd'] for r in requests if r['label'].endswith('/judge')),
        judge_format_errors=sum(r['status'] == 'format_error' for r in records),
        judge_deferred=sum(r.get('decision') == 'defer' for r in records),
        rule_overrides=sum(bool(r.get('judge', {}).get('changed_rule_decision')) for r in trace),
        actual_tail_interruptions=sum(bool(r.get('actual_discarded_actions')) and not r['done']
                                      and bool(r.get('interruption_reason')) for r in trace),
        judge_latency_seconds=sum(r['latency_seconds'] for r in records),
        prompt_tokens=row['prompt_tokens'] + sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in records),
        completion_tokens=row['completion_tokens'] + sum((r.get('usage') or {}).get('completion_tokens', 0) for r in records),
        total_tokens=row['total_tokens'] + sum((r.get('usage') or {}).get('total_tokens', 0) for r in records))
    save(folder / 'result.json', row)
    save(directory / 'result.json', row)
    return row

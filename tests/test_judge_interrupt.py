from dataclasses import replace
import json
from types import SimpleNamespace
import pytest

from dynamic_skill.controllers.judge import JudgeConfig, JudgeExecutor, LocalJudge, parse_verdict
from dynamic_skill.controllers.control_only_evolution import ControlPolicy
from dynamic_skill.infrastructure.campaign_budget import CampaignHalt
from experiments.alfworld.judge_interrupt.runtime import EpisodeAllowance, SharedBudgetJudge


def setup(decision, **kwargs):
    seen = []
    def request(prompt, **options):
        seen.append(json.loads(prompt))
        return {'response': json.dumps({'decision': decision, 'evidence': 'Observed dependency.'}),
                'finish_reason': 'stop', 'cost_usd': .01}
    judge = LocalJudge(request, JudgeConfig(**kwargs))
    ctl = JudgeExecutor(ControlPolicy(), dynamic=True, judge=judge)
    ctl.start('Room.', 'Put a book on desk.')
    return ctl, judge, seen


def step(ctl, queue, observation='Nothing happens.', done=False):
    ctl.load_chunk(queue)
    action = ctl.next_action()
    ctl.before_action(action, 'Room.', ['go to desk 1'])
    ctl.after_action(['go to desk 1'])
    return ctl.observe(action, observation, done=done)


def test_judge_can_preserve_independent_tail_after_failed_action():
    ctl, judge, seen = setup('continue')
    event = step(ctl, ['take book 1 from table 1', 'go to desk 1'])
    assert list(ctl.pending_actions) == ['go to desk 1']
    assert not event.discarded_actions and len(seen) == 1
    assert ctl.last_execution_audit['judge']['changed_rule_decision']
    assert set(seen[0]) == {'task', 'last_action', 'observation', 'remaining_queue',
        'admissible_now', 'new_admissible_commands', 'recent_observations', 'decisions_so_far'}
    assert 'reward' not in seen[0] and 'rule_should_interrupt' not in seen[0]


def test_judge_interrupt_changes_real_queue_and_recovery_reason():
    ctl, _, _ = setup('interrupt')
    event = step(ctl, ['go to table 1', 'take book 1 from table 1'], 'You arrive at table 1.')
    assert event.discarded_actions == ('take book 1 from table 1',)
    assert ctl.needs_plan and ctl.last_replan_reason.startswith('judge interrupt:')


@pytest.mark.parametrize('decision', ['defer', 'invalid'])
def test_abstention_and_malformed_verdict_keep_rule_and_charge(decision):
    ctl, judge, _ = setup(decision)
    event = step(ctl, ['take book 1 from table 1', 'go to desk 1'])
    assert event.discarded_actions == ('go to desk 1',)
    assert len(judge.records) == 1 and judge.records[0]['cost_usd'] == .01


@pytest.mark.parametrize('queue,done', [(['look'], False), (['look', 'go to desk 1'], True)])
def test_no_judge_on_empty_tail_or_terminal(queue, done):
    ctl, judge, _ = setup('interrupt')
    step(ctl, queue, done=done)
    assert judge.records == []


def test_no_judge_for_ordinary_valid_queue_and_no_false_later_precondition_gate():
    ctl, judge, _ = setup('interrupt')
    step(ctl, ['look', 'go to desk 1', 'move book 1 to desk 1'], 'A room.')
    assert not judge.records and list(ctl.pending_actions) == ['go to desk 1', 'move book 1 to desk 1']


def test_check_budget_and_context_cap_fall_back_without_requests():
    ctl, judge, _ = setup('defer', max_calls=1, max_context_chars=1000)
    assert judge.decide({'long': 'x' * 1001})['status'] == 'context_cap'
    step(ctl, ['take book 1 from table 1', 'go to desk 1'])
    step(ctl, ['take book 2 from table 1', 'go to desk 1'])
    assert len(judge.records) == 1
    assert ctl.last_execution_audit['judge']['status'] == 'call_cap'


def test_transport_halts_are_not_silently_hidden():
    def fail(*a, **kw): raise CampaignHalt('budget_stop')
    judge = LocalJudge(fail, JudgeConfig())
    with pytest.raises(CampaignHalt): judge.decide({})
    assert judge.records[0]['status'] == 'request_error'


def test_actor_and_judge_share_total_allowance_and_keep_recovery_call():
    replies = []
    def request(*a, **kw):
        replies.append(kw)
        return {'response': '{"decision":"defer","evidence":"Unknown."}', 'finish_reason':'stop'}
    transport = SimpleNamespace(request=request, settings=SimpleNamespace(max_tokens=512))
    allowance = EpisodeAllowance(transport, '/trial', 3)
    judge = SharedBudgetJudge(allowance, JudgeConfig(), lambda rows: None)
    allowance('plan')
    assert judge.decide({})['decision'] == 'defer' and allowance.remaining == 1
    assert judge.decide({})['status'] == 'shared_call_budget'
    allowance('recovery')
    with pytest.raises(CampaignHalt, match='model_call_limit'): allowance.before_call()
    assert len(replies) == 3 and [r['label'] for r in replies] == ['/trial/actor','/trial/judge','/trial/actor']


def test_truncated_json_is_not_a_valid_judgment():
    with pytest.raises(ValueError):
        parse_verdict({'response':'{"decision":"continue","evidence":"OK"}', 'finish_reason':'length'})


def test_hard_public_fact_blocks_llm_override_without_spending():
    from dynamic_skill.controllers.judge import HardEvidenceExecutor
    def forbidden(*a, **kw):
        raise AssertionError('A known impossible next action needs no model call')
    judge = LocalJudge(forbidden, JudgeConfig())
    baseline = HardEvidenceExecutor(ControlPolicy(), dynamic=True, authoritative_admissibility=True)
    judged = JudgeExecutor(ControlPolicy(), dynamic=True, judge=judge, authoritative_admissibility=True)
    for ctl in (baseline, judged):
        ctl.start('Room.', 'Put a book on desk.')
        event = step(ctl, ['look', 'take book 1 from table 1'], 'A room.')
        assert event.discarded_actions == ('take book 1 from table 1',)
        assert ctl.last_execution_audit['hard_head_guard']
    assert judge.records == [] and judged.judge_audit['skip'] == 'hard_public_evidence'


def test_hard_guard_never_checks_future_actions_against_current_commands():
    from dynamic_skill.controllers.judge import HardEvidenceExecutor
    ctl = HardEvidenceExecutor(ControlPolicy(), dynamic=True, authoritative_admissibility=True)
    ctl.start('Room.', 'Put a book on desk.')
    step(ctl, ['look', 'go to desk 1', 'move book 1 to desk 1'], 'A room.')
    assert list(ctl.pending_actions) == ['go to desk 1', 'move book 1 to desk 1']
    assert not ctl.last_execution_audit['hard_head_guard']


def test_partial_action_hints_do_not_authorize_a_hard_veto():
    from dynamic_skill.controllers.judge import HardEvidenceExecutor
    ctl = HardEvidenceExecutor(ControlPolicy(), dynamic=True, authoritative_admissibility=False)
    ctl.start('Room.', 'Put a book on desk.')
    step(ctl, ['look', 'take book 1 from table 1'], 'A room.')
    assert list(ctl.pending_actions) == ['take book 1 from table 1']


def test_real_runner_rule_prompt_parity_and_judge_costs(tmp_path):
    from experiments.alfworld.joint_evolution.demo import PublicFixture
    from experiments.alfworld.joint_evolution.runtime import evaluate_episode
    from experiments.alfworld.joint_evolution.schema import Genome
    from experiments.alfworld.judge_interrupt.runtime import evaluate
    from experiments.common.puzzle_pilot import Settings
    settings = Settings(model='fixture', max_steps=6, max_calls=3)
    task = {'id':'fixture', 'gamefile':'offline-fixture://fixture'}
    genome = Genome('Go to the desk.')

    class Transport:
        def __init__(self):
            self.settings = settings
            self.records, self.prompts = [], []
            self.budget = SimpleNamespace(read=lambda: {'requests': self.records})
            self.count = 0
        def request(self, prompt, *, label, **kwargs):
            self.records.append({'label':label, 'budget_charge_usd':.01})
            if label.endswith('/judge'):
                reply = '{"decision":"continue","evidence":"Navigation is independent."}'
            else:
                self.prompts.append(prompt)
                self.count += 1
                actions = ['open drawer 1', 'go to desk 1'] if self.count == 1 else ['go to desk 1']
                reply = '<thought>Navigate.</thought><actions>' + json.dumps(actions) + '</actions>'
            return {'response':reply, 'finish_reason':'stop',
                    'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}, 'cost_usd':.01}
    original = Transport()
    actor = EpisodeAllowance(original, '/original', 3)
    old = evaluate_episode(PublicFixture(), task, genome, settings, actor, tmp_path/'old')
    baseline = Transport()
    new = evaluate(PublicFixture(), task, genome, baseline, tmp_path/'new', 'rules', {})
    assert baseline.prompts == original.prompts
    assert new['model_calls'] == old['model_calls'] == 2
    judged = evaluate(PublicFixture(), task, genome, Transport(), tmp_path/'judge', 'dependency', {})
    assert judged['success'] and judged['actor_calls'] == judged['judge_calls'] == 1
    assert judged['total_model_calls'] == 2 and judged['cost_usd'] == .02
    assert judged['total_tokens'] == 30 and judged['actual_tail_interruptions'] == 0
    assert judged['rule_overrides'] == 1

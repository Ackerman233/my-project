from dynamic_skill.paper_control import ValueExecutor
from dynamic_skill.control_only_evolution import ControlPolicy
from dynamic_skill.value_termination import features,fit
from dynamic_skill.interrupter import DecisionType


def test_supported_positive_branch_value_discards_only_actual_pending_tail():
    task='put a mug in desk.';obs='You arrive at shelf 1. On the shelf 1, you see nothing.'
    after=['go to desk 1'];history=[{'action':'go to shelf 1','env_feedback':obs}]
    x=features(task,obs,after,history,['go to desk 1'])
    model=fit([{'features':x,'utilities':[.1,.9],'group':str(i)} for i in range(4)])
    ctl=ValueExecutor(ControlPolicy(max_actions=4),model);ctl.start('Room with shelf 1 and desk 1.',task)
    ctl.load_chunk(['go to shelf 1','go to desk 1']);action=ctl.next_action()
    ctl.before_action(action,'Room.',['go to shelf 1']);ctl.after_action(after)
    event=ctl.observe(action,obs,reward=0.,done=False,info={})
    assert event.decision.kind is DecisionType.REPLAN
    assert event.discarded_actions==('go to desk 1',) and ctl.needs_plan
    assert ctl.value_interrupts==ctl.interruption_count==1
    assert ctl.last_execution_audit['actual_discarded_actions']==['go to desk 1']


def test_terminal_success_is_never_overridden_by_value_termination():
    x=features('put a mug in desk.','Success.',[],[],['go to shelf 1'])
    model=fit([{'features':x,'utilities':[.1,.9],'group':str(i)} for i in range(4)])
    ctl=ValueExecutor(ControlPolicy(),model);ctl.start('Room.','put a mug in desk.')
    ctl.load_chunk(['move mug 1 to desk 1','go to shelf 1']);action=ctl.next_action()
    ctl.before_action(action,'Room.',['move mug 1 to desk 1']);ctl.after_action([])
    event=ctl.observe(action,'You move the mug 1 to the desk 1.',reward=1.,done=True,info={'won':True})
    assert event.decision.kind is DecisionType.STOP_SUCCESS
    assert ctl.value_interrupts==0 and not ctl.value_decisions

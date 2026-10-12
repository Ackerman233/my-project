"""Optional learned termination, with existing hard checks preserved."""
from dataclasses import replace
from dynamic_skill.controllers.control_only_evolution import ControlExecutor
from dynamic_skill.execution.interrupter import Decision,DecisionType
from dynamic_skill.controllers.value_termination import features,predict


class ValueExecutor(ControlExecutor):
    def __init__(self,policy,model):
        super().__init__(policy,dynamic=True,state_check=True)
        self.value_model=model;self.value_history=[];self.value_decisions=[];self.value_interrupts=0
    def observe(self,*args,**kwargs):
        action,observation=args[:2]
        event=super().observe(*args,**kwargs)
        self.value_history.append({'action':action,'env_feedback':observation})
        decision=None
        if self.pending_actions and event.decision.kind is DecisionType.CONTINUE:
            x=features(self.monitor.task,observation,self.admissible_after,self.value_history,list(self.pending_actions))
            decision=predict(self.value_model,x)
            decision.update(step=len(self.value_history)-1,features=x)
            self.value_decisions.append(decision)
            if decision['terminate'] and self.anomaly_replan_count<self.interrupter.max_interrupts:
                discarded=tuple(self.pending_actions);self.pending_actions.clear()
                self.interruption_count+=1;self.anomaly_replan_count+=1;self.value_interrupts+=1
                reason='Learned branch utility favors replanning after API/action costs; advantage='+format(decision['advantage'],'.4f')
                self.last_replan_reason=reason
                self.plan=self.inducer.replan(self.plan,self.active_index,reason,observation);self.active_index=0
                event=replace(event,decision=Decision(DecisionType.REPLAN,reason),discarded_actions=discarded,
                    plan_revision=self.plan.revision,interruption_count=self.interruption_count)
                self.last_execution_audit.update(actual_discarded_actions=list(discarded),interruption_reason=reason,
                    interrupt_count=self.interruption_count,anomaly_replans=self.anomaly_replan_count)
        self.last_execution_audit.update(value_estimate=decision,value_interrupts=self.value_interrupts)
        return event

"""V2: evolve executable controls only; task instructions remain byte-identical."""
from dataclasses import dataclass,asdict,replace
import hashlib,json
from dynamic_skill.execution.checker import ObservationChecker,CheckStatus,CheckResult
from dynamic_skill.execution.interrupter import DecisionType
from dynamic_skill.execution.skill_executor import ChunkExecutor
from dynamic_skill.alfworld.acquisition_check import AcquisitionMonitor
from dynamic_skill.alfworld.alfworld_recovery import recovery_prompt

@dataclass(frozen=True)
class ControlPolicy:
    max_actions:int=4
    trigger:str='anomaly'
    invalid_patience:int=1
    recovery_steps:int=0

    def __post_init__(self):
        for key,low,high in [('max_actions',1,4),('invalid_patience',1,2),('recovery_steps',0,3)]:
            value=getattr(self,key)
            if type(value) is not int or not low<=value<=high:raise ValueError(key)
        if self.trigger not in ('invalid','anomaly'):raise ValueError('trigger')

def fingerprint(text):return hashlib.sha256(text.encode()).hexdigest()

def accept_proposal(payload,baseline,task_skill):
    """Validation of a proposal is NOT acceptance on independent tasks."""
    if isinstance(payload,str):payload=json.loads(payload)
    if not isinstance(payload,dict) or set(payload)!={'task_skill_sha256','control'}:
        raise ValueError('only a task-skill hash and control parameters are permitted')
    if payload['task_skill_sha256']!=fingerprint(task_skill):raise ValueError('task skill changed')
    if not isinstance(payload['control'],dict) or set(payload['control'])!=set(asdict(baseline)):
        raise ValueError('control schema; prose edits and recovery_note are forbidden')
    candidate=ControlPolicy(**payload['control'])
    changed=[k for k,v in asdict(baseline).items() if asdict(candidate)[k]!=v]
    return {'policy':candidate,'changed_fields':changed,'status':'candidate' if changed else 'noop',
            'task_skill':task_skill,'task_skill_sha256':fingerprint(task_skill)}

def neighbors(baseline):
    """Bounded one-coordinate search. Include incumbent, never force changes."""
    result=[baseline]
    for key,values in {'max_actions':(1,2,3,4),'trigger':('invalid','anomaly'),
                       'invalid_patience':(1,2),'recovery_steps':(0,1,2,3)}.items():
        for value in values:
            if value!=getattr(baseline,key):result.append(replace(baseline,**{key:value}))
    return result

class PolicyChecker(ObservationChecker):
    def __init__(self,policy):
        super().__init__();self.policy=policy;self.invalid_streak=0;self.semantic=None;self.discovery=None
    def check(self,*args,**kwargs):
        result=super().check(*args,**kwargs)
        self.invalid_streak=self.invalid_streak+1 if result.status is CheckStatus.INVALID_ACTION else 0
        if self.discovery and self.discovery['triggered'] and result.status is CheckStatus.OK and not result.should_interrupt:
            return CheckResult(CheckStatus.OK,'target discovery: '+self.discovery['evidence_command'],True,('target_discovery',))
        if (self.semantic and self.semantic['blocks_tail'] and
                result.status not in (CheckStatus.SUCCESS,CheckStatus.ENV_FAILURE) and not result.should_interrupt):
            return CheckResult(CheckStatus.OK, 'acquisition '+self.semantic['status']+': '+self.semantic['reason']+
                               '; remaining action requires '+self.semantic['expected']['object'], True,
                               ('acquisition_contract',self.semantic['status']))
        if result.status is CheckStatus.INVALID_ACTION:
            return replace(result,should_interrupt=self.invalid_streak>=self.policy.invalid_patience)
        if self.policy.trigger=='invalid' and result.status is not CheckStatus.ENV_FAILURE:
            return replace(result,should_interrupt=False)
        return result

class ControlExecutor(ChunkExecutor):
    def __init__(self,policy,*,dynamic=True,state_check=False,discovery_interrupt=False):
        super().__init__(chunk_size=policy.max_actions,dynamic=dynamic,checker=PolicyChecker(policy))
        self.policy=policy;self.recovery_remaining=0
        self.state_check=state_check;self.monitor=AcquisitionMonitor()
        self.original_queue=[];self.before={};self.last_execution_audit={};self.failures=[]
        self.planning_context={}
        self.discovery_interrupt=discovery_interrupt;self.admissible_after=[]
    def after_action(self,admissible):
        self.admissible_after=list(admissible)
    def start(self,initial_observation,task=None):
        plan=super().start(initial_observation,task)
        self.monitor.task=plan.task
        return plan
    def before_action(self,action,observation,admissible):
        self.before={'observation':observation,'admissible':list(admissible)}
    @property
    def planning_limit(self):return 1 if self.recovery_remaining else self.policy.max_actions
    def load_chunk(self,actions):
        if len(actions)>self.planning_limit:raise ValueError('exceeds current recovery action limit')
        result=super().load_chunk(actions)
        self.original_queue=list(actions)
        return result
    def observe(self,*args,**kwargs):
        action,observation=args[:2];step=len(self.action_history)
        remaining=list(self.pending_actions);counter_before=self.recovery_remaining
        check=self.monitor.check(action,observation,remaining,step,self.before)
        from dynamic_skill.alfworld.space_boundary import discovery_check
        discovery=discovery_check(self.monitor.task,action,remaining,self.admissible_after,step)
        self.checker.discovery=discovery if self.discovery_interrupt else None
        self.checker.semantic=check if self.state_check else None
        # Only fresh matching evidence can resolve a tracked acquisition failure.
        if check.get('applicable') and check['status']=='confirmed':
            for failure in self.failures:
                if failure['object']==check['expected']['object'] and step>failure['step'] and failure['resolution']=='unverified':
                    failure.update(resolution='verified_resolved',resolution_evidence=check['evidence'])
        event=super().observe(*args,**kwargs)
        if event.decision.kind is DecisionType.REPLAN:
            self.recovery_remaining=self.policy.recovery_steps
            if not event.decision.reason.startswith('target discovery:'):
                self.failures.append({'failure_id':len(self.failures)+1,'step':step,'action':action,
                    'reason':event.decision.reason,'object':check.get('expected',{}).get('object'),
                    'resolution':'unverified','resolution_evidence':None})
        elif event.check.status not in (CheckStatus.INVALID_ACTION,CheckStatus.ACTION_LOOP,CheckStatus.STALLED,CheckStatus.ENV_FAILURE):
            self.recovery_remaining=max(0,self.recovery_remaining-1)
        self.last_execution_audit={'original_action_queue':list(self.original_queue),
            'action_position':self.actions_in_chunk-1,'remaining_before_interrupt':remaining,
            'actual_discarded_actions':list(event.discarded_actions),
            'state_check_enabled':self.state_check,'state_check':check,
            'discovery_check':discovery,'discovery_interrupt_enabled':self.discovery_interrupt,
            'discovery_pause':event.decision.kind is DecisionType.REPLAN and event.decision.reason.startswith('target discovery:'),
            'semantic_pause':bool(self.dynamic and self.state_check and check['blocks_tail'] and
                                   event.decision.kind is DecisionType.REPLAN),
            'interruption_reason':event.decision.reason if event.decision.kind is DecisionType.REPLAN else None,
            'recovery_remaining_before':counter_before,'recovery_remaining_after':self.recovery_remaining,
            'recovery_counter_ended':counter_before>0 and self.recovery_remaining==0,
            'failure_resolution':json.loads(json.dumps(self.failures)),
            'original_failure_verified_resolved':self.failures[-1]['resolution']=='verified_resolved' if self.failures else None,
            'anomaly_replans':self.anomaly_replan_count,'interrupt_count':self.interruption_count}
        return event

def run_control_episode(env,task,settings,output,planner,task_skill,policy,*,dynamic=True,
                        history_evidence=False,state_check=False,boundary_guidance=False,discovery_interrupt=False,macro_mode=False):
    """Existing real ALFWorld runner, with a dynamic planning cap enforced twice."""
    from dataclasses import replace as dc_replace
    from experiments.alfworld_campaign import run_episode
    from experiments.puzzle_pilot import save
    from dynamic_skill.alfworld.micro_loop_protocol import make_prompt,parse_response,SYSTEM
    executor=ControlExecutor(policy,dynamic=dynamic,state_check=state_check,discovery_interrupt=discovery_interrupt)
    limits=[]
    current_context={}
    def prompt(ignored,**context):
        current_context.clear();current_context.update(context)
        # Controls are implemented in code, not inserted as editable task prose.
        limit=min(executor.planning_limit,settings.max_steps-len(context['trace']))
        limits.append(limit)
        body=make_prompt(task_skill,limit,'Replan from the current observation.')(ignored,**context)
        if boundary_guidance:
            from dynamic_skill.alfworld.space_boundary import boundary_prompt
            body=boundary_prompt(body)
        if macro_mode:
            from dynamic_skill.alfworld.space_subskills import INSTRUCTIONS
            body+=INSTRUCTIONS
        enhanced=bool(history_evidence and context['anomaly'])
        if enhanced:
            body=recovery_prompt(body,**context,output_protocol='thought_actions')
        recovery_info=None
        if state_check and context['anomaly']:
            recovery_info={'last_check':json.loads(json.dumps(executor.last_execution_audit.get('state_check'))),
                           'failures':json.loads(json.dumps(executor.failures)),
                           'planning_at_step':len(context['trace']),
                           'note':'A recovery counter reaching zero is not evidence the failed condition is resolved. '
                                  'If possession is unknown, inventory is a real environment action to verify it.'}
            for evidence in ([recovery_info['last_check'].get('evidence')] if recovery_info['last_check'] else []) + [
                    f.get('resolution_evidence') for f in recovery_info['failures']]:
                if evidence:
                    evidence.update(age_steps=len(context['trace'])-1-evidence['observed_at_step'],historical=True,
                                    current_state_guaranteed=False)
            body+='\nAcquisition recovery context:\n'+json.dumps(recovery_info)
        executor.planning_context={'generic_anomaly_prompt_used':bool(context['anomaly']),
            'historical_evidence_injected':enhanced,'recovery_prompt_used':enhanced,
            'single_step_recovery_active':executor.recovery_remaining>0,
            'recovery_remaining':executor.recovery_remaining,'effective_action_limit':limit,
            'anomaly_received':context['anomaly'],'state_recovery_received':recovery_info,
            'failure_resolution':json.loads(json.dumps(executor.failures))}
        return body
    def parse(text,limit):
        limit=min(limit,executor.planning_limit)
        normalized=False
        if macro_mode:
            from dynamic_skill.alfworld.space_subskills import normalize_response
            text,normalized=normalize_response(text)
        actions,reason=parse_response(text,limit)
        executor.response_audit={}
        if macro_mode:
            from dynamic_skill.alfworld.space_subskills import expand
            actions,audit=expand(actions,current_context,limit)
            executor.response_audit={'subskill_expansion':audit,'model_native_actions':parse_response(text,limit)[0],
                                     'protocol_normalized':normalized,'normalization_kind':'missing_actions_tags' if normalized else None}
        return actions,reason
    if macro_mode:
        from dynamic_skill.alfworld.space_subskills import SYSTEM_SKILL
        SYSTEM=SYSTEM_SKILL
    row=run_episode(env,task,'dynamic_interrupt' if dynamic else 'fixed_chunk',
        dc_replace(settings,chunk_size=policy.max_actions),output,planner,task_skill,
        prompt_transform=prompt,response_parser=parse,system_prompt=SYSTEM,executor=executor)
    path=output/'predictions'/row['id']
    calls=[]
    if (path/'model_calls.json').exists():
        calls=json.loads((path/'model_calls.json').read_text())
        for call,limit in zip(calls,limits):call['effective_action_limit']=limit
        save(path/'model_calls.json',calls)
    trace=json.loads((path/'conversation.json').read_text()) if (path/'conversation.json').exists() else []
    row.update(control_policy=asdict(policy),task_skill_sha256=fingerprint(task_skill),
               protocol='control-only-v3' if history_evidence or state_check else 'control-only-v2',
               history_evidence_enabled=history_evidence,state_check_enabled=state_check,
               boundary_guidance=boundary_guidance,discovery_interrupt=discovery_interrupt,
               macro_mode=macro_mode,subskill_calls=sum(bool(c.get('subskill_expansion')) for c in calls),
               discovery_pauses=sum(r.get('discovery_pause',False) for r in trace),
               effective_action_limits=limits[:row['model_calls']],
               history_evidence_calls=sum(c.get('historical_evidence_injected',False) for c in calls),
               single_step_recovery_calls=sum(c.get('single_step_recovery_active',False) for c in calls),
               semantic_pauses=sum(r.get('semantic_pause',False) for r in trace),
               state_check_exposures=sum(r.get('state_check',{}).get('applicable',False) for r in trace) if state_check else 0,
               recovery_counter_end_events=sum(r.get('recovery_counter_ended',False) for r in trace),
               failure_resolution=executor.failures,
               false_interrupt_count=None,
               false_interrupt_note='Requires labeled paired/counterfactual review; unknown is not a false positive label.')
    save(path/'result.json',row)
    return row

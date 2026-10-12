"""简短显式决策与原生思考分离；有界可执行技能参数。"""
import json,re
from dataclasses import replace
from dynamic_skill.alfworld_rollout import parse_action_chunk
from dynamic_skill.checker import ObservationChecker,CheckStatus

SYSTEM=('Operate ALFWorld using the provided skill and observations. '
        'Return <thought>a brief decision summary, at most two sentences, no detailed chain of thought</thought>'
        '<actions>["concrete command"]</actions>. Respect the requested action count. '
        'A multi-action queue executes sequentially without a model call between commands. '
        'Do not output native thinking tags, programs, or markdown.')
INITIAL_CONTROL={'max_actions':4,'trigger':'anomaly','recovery_note':'Replan from the latest observation; do not repeat rejected actions without a changed precondition.'}
def controller_block(c):return '<controller_json>'+json.dumps(c)+'</controller_json>'
def parse_controller(skill):
    matches=re.findall(r'<controller_json>(.*?)</controller_json>',skill,re.S)
    if len(matches)!=1:raise ValueError('must preserve exactly one controller block')
    c=json.loads(matches[0])
    if set(c)!=set(INITIAL_CONTROL):raise ValueError('controller schema')
    if type(c['max_actions']) is not int or not 1<=c['max_actions']<=4:raise ValueError('chunk bounds')
    if c['trigger'] not in ['invalid','anomaly']:raise ValueError('trigger bounds')
    if not isinstance(c['recovery_note'],str) or len(c['recovery_note'])>1200:raise ValueError('note bounds')
    return c
def parse_response(text,limit):
    m=re.fullmatch(r'\s*<thought>([^<]+)</thought>\s*(<actions>.*?</actions>)\s*',text,re.S)
    if not m:raise ValueError('thought/actions protocol')
    return parse_action_chunk(m[2],limit),m[1].strip()
class SkillChecker(ObservationChecker):
    def __init__(self,trigger):super().__init__();self.trigger=trigger
    def check(self,*args,**kwargs):
        r=super().check(*args,**kwargs)
        if self.trigger=='invalid' and r.status not in [CheckStatus.INVALID_ACTION,CheckStatus.ENV_FAILURE]:
            return replace(r,should_interrupt=False)
        return r

def make_prompt(skill,limit,note=''):
    def prompt(_ignored,*,task,initial,observation,admissible,trace,anomaly):
        history=[{k:r.get(k,'') for k in ['action','reasoning','env_feedback']} for r in trace[-8:]]
        body=(skill+'\nTask: '+task+'\nRecent Thought/Action/Observation history:\n'+json.dumps(history)
              +'\nCurrent observation:\n'+observation+'\nAdmissible now:\n'+json.dumps(admissible)
              +f'\nReturn 1 to {limit} concrete actions. First action must be currently admissible. '
              'Later actions must be valid after preceding commands. Stop before unknown information. Never invent object IDs. '
              'Return the specified thought/actions blocks.')
        if anomaly:body+='\nObserved anomaly: '+anomaly+'\nRecovery policy: '+note
        return body
    return prompt

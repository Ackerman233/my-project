"""ALFWorld-specific reusable frontier skill, from public feedback/commands only."""
import re
from dynamic_skill.alfworld.space_subskills import CONTAINERS
from dynamic_skill.alfworld.acquisition_check import inventory_objects

ENTITY=r'[a-z][a-z0-9]* [0-9]+'


def task_target(task):
    match=re.match(r'(?:put|examine|look at) (?:(?:some|an|a|the) )?([a-z]+)\b',task.lower().strip())
    return match[1] if match else None


class ObservableSearch:
    def __init__(self,task,*,acquire=False,priority='surface_first'):
        self.task,self.target=task,task_target(task)
        self.light=bool(re.match(r'(?:examine|look at) ',task.lower().strip()))
        self.acquire,self.priority=acquire,priority
        self.searched=set();self.objects={};self.held=None;self.current=None;self.last={};self.step=-1
    def update(self,action,feedback,admissible,step):
        self.step=step
        arrived=re.search(r'You arrive at (?:the )?('+ENTITY+r')\.',feedback)
        if arrived and action=='go to '+arrived[1]:self.current=arrived[1]
        loc=None
        surface=re.search(r'On the ('+ENTITY+r'), you see (.*)',feedback)
        if surface:loc=surface[1]
        elif 'In it, you see ' in feedback:
            if action.startswith(('go to ','open ')):
                loc=action.split(' ',2)[-1] if action.startswith('go to ') else action[5:]
        if loc:
            self.searched.add(loc)
            # A current observation supersedes historical contents at this exact location.
            self.objects={obj:record for obj,record in self.objects.items() if record['location']!=loc}
            contents=surface[2] if surface else feedback.split('In it, you see ',1)[1]
            for obj in re.findall(r'\b('+ENTITY+r')\b',contents):
                self.objects[obj]={'location':loc,'observed_step':step}
        pickup=re.fullmatch(r'You pick up the ('+ENTITY+r') from the ('+ENTITY+r')\.',feedback)
        moved=re.fullmatch(r'You move the ('+ENTITY+r') to the ('+ENTITY+r')\.',feedback)
        if action=='inventory':
            inventory=inventory_objects(feedback)
            if inventory is not None:self.held=inventory[0] if len(inventory)==1 else None
        if pickup:
            self.held=pickup[1];self.objects.pop(pickup[1],None)
        elif moved:
            if self.held==moved[1]:self.held=None
            self.objects[moved[1]]={'location':moved[2],'observed_step':step}
        self.last={'feedback':feedback,'admissible':list(admissible)}
    def action(self,observation,admissible):
        if not self.target:return None
        commands=set(admissible)
        holding_target=bool(self.held and self.held.rsplit(' ',1)[0]==self.target)
        if holding_target and not self.light:return None
        wanted='desklamp' if holding_target and self.light else self.target
        if wanted==self.target:
            available=[]
            for a in admissible:
                match=re.fullmatch(r'take ('+re.escape(wanted)+r' [0-9]+) from ('+ENTITY+r')',a)
                if match:available.append(a)
            if available:
                return {'actions':[sorted(available)[0]],'kind':'certified_acquire','wanted':wanted} if self.acquire else None
        else:
            if any(re.fullmatch(r'use desklamp [0-9]+',a) for a in admissible):return None
        known=[r for obj,r in self.objects.items() if obj.rsplit(' ',1)[0]==wanted]
        for record in sorted(known,key=lambda r:(-r['observed_step'],r['location'])):
            nav='go to '+record['location']
            if nav in commands:
                return {'actions':[nav],'kind':'verify_cached_location','wanted':wanted,
                    'historical_evidence_age':self.step-record['observed_step'],'current_object_guaranteed':False}
            if wanted=='desklamp' and record['location']==self.current:return None
        if self.current and 'open '+self.current in commands and self.current not in self.searched:
            return {'actions':['open '+self.current],'kind':'fresh_supported_open','wanted':wanted}
        frontier=[a for a in admissible if a.startswith('go to ') and a[6:] not in self.searched]
        def key(action):
            loc=action[6:];kind,num=loc.rsplit(' ',1)
            return (kind in CONTAINERS,int(num),kind) if self.priority=='surface_first' else (int(num),kind)
        if frontier:
            return {'actions':[min(frontier,key=key)],'kind':'unsearched_frontier','wanted':wanted}
        return None
    def public_state(self):
        return {'held':self.held,'current_location':self.current,'searched':sorted(self.searched),
            'historical_objects':self.objects,'source':'recorded public feedback and current exact admissible commands'}

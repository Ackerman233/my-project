"""恢复时补充可观察证据，不查询环境隐藏状态、规划器或答案。"""
import json
from dynamic_skill.execution.checker import ObservationChecker

def recovery_prompt(prompt, *, task, initial, observation, admissible, trace, anomaly,
                    output_protocol='actions_only'):
    if output_protocol not in ('actions_only', 'thought_actions'):
        raise ValueError('unsupported recovery output protocol')
    if not anomaly:
        return prompt
    def rejected_row(row):
        return (not row.get('valid',True) or (output_protocol=='thought_actions' and
                (row.get('checker_status')=='invalid_action' or any(
                    p in row['env_feedback'].lower() for p in ObservationChecker.invalid_patterns))))
    # 原始观测保留来源，不把历史可见物体断言为现在仍在原位置。
    discoveries=[]
    for index,row in enumerate(trace):
        text=row['env_feedback']
        if row.get('valid') and not rejected_row(row) and any(w in text.lower() for w in ('you arrive','you open','you see')):
            discoveries.append({'step':index,'action':row['action'],'observed_then':text[:700]})
    rejected=[{'step':i,'action':r['action'],'feedback':r['env_feedback'][:300]}
              for i,r in enumerate(trace) if rejected_row(r)]
    changes=[{'step':i,'action':r['action'],'feedback':r['env_feedback'][:300]}
             for i,r in enumerate(trace) if r.get('valid') and not rejected_row(r) and any(w in r['env_feedback'].lower()
                 for w in ('you pick up','you move','you put','you clean','you heat','you cool','you turn on'))]
    packet={'task':task,'trigger':anomaly,'initial_observation':initial[:1400],
            'historical_location_observations':discoveries[-6:],
            'recent_confirmed_changes':changes[-6:],'rejected_actions':rejected[-6:]}
    if output_protocol == 'thought_actions':
        packet['current_observation']={'text':observation,'source':'latest env feedback','step':len(trace)-1}
        packet['initial_observation_metadata']={'source':'env.reset','age_steps':len(trace),'historical':True}
        for key in ('historical_location_observations','recent_confirmed_changes','rejected_actions'):
            for item in packet[key]:
                item.update(source='recorded env.step feedback',age_steps=len(trace)-1-item['step'],
                            historical=True,current_state_guaranteed=False)
        # "valid" and generic change strings are retrieval filters, not a semantic proof.
        packet['retrieval_note']='Retrieved change reports are historical evidence, not verified current possession or goal completion.'
    contract = ('Return only the same actions JSON block; no additional explanation.'
                if output_protocol == 'actions_only' else
                'Keep the same <thought>brief decision summary</thought><actions>JSON string array</actions> protocol. '
                'Respect the current action limit; do not add other output blocks.')
    return prompt+'\nRecovery evidence (historical observations, not current-state guarantees):\n'+json.dumps(packet)+(
        '\nRe-evaluate the next unfinished subgoal using the task and current observation. '
        'Do not repeat a rejected action unless its missing precondition has changed. '
        'Use confirmed state changes to track what you hold and where you have already searched. '
        'Distinguish inspecting an object from satisfying a task-specific operation. '
        'Choose a currently admissible first action; if needed, gather missing information. '
        + contract)

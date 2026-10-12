"""Narrow observable contract: acquire a task-mentioned, exact ALFWorld object.

No SkillPlan completion inference, hidden state, solver, or extra environment calls.
Only exact feedback grammars are evidence; other text remains unknown.
"""
import re

ENTITY = r'[a-z][a-z0-9]* [0-9]+'
TAKE = re.compile(rf'take ({ENTITY}) from ({ENTITY})')
PICKUP = re.compile(rf'you pick up the ({ENTITY}) from the ({ENTITY})\.')

def norm(value):
    return ' '.join(value.lower().split())

def take_target(action):
    match = TAKE.fullmatch(norm(action))
    return match.groups() if match else None

def task_mentions(task, obj):
    return bool(re.search(r'\b' + re.escape(obj.rsplit(' ', 1)[0]) + r's?\b', norm(task)))

def dependencies(obj, remaining):
    """Stop lookahead at explicit re-acquisition or inventory verification.

    Only these exact commands have supported possession prerequisites. Do not
    infer lamp/use semantics or block an unrelated navigation-only tail.
    """
    found = []
    for index, raw in enumerate(remaining):
        action = norm(raw)
        take = take_target(action)
        if (take and take[0] == obj) or action == 'inventory':
            break
        if re.fullmatch(rf'(move {re.escape(obj)} to|(?:clean|heat|cool) {re.escape(obj)} with) {ENTITY}', action):
            found.append({'remaining_index': index, 'action': raw,
                          'requires': 'holding(' + obj + ')'})
    return found

def inventory_objects(observation):
    text = norm(observation)
    if text == 'you are not carrying anything.':
        return []
    match = re.fullmatch(r'you are carrying: (.+)\.', text)
    if not match:
        return None
    parts = re.split(r',\s*(?:and )?| and ', match[1])
    objects = []
    for part in parts:
        item = re.fullmatch(rf'(?:a |an |the )?({ENTITY})', part)
        if not item:
            return None  # Never infer absence from an incompletely parsed list.
        objects.append(item[1])
    return objects

class AcquisitionMonitor:
    def __init__(self):
        self.task = ''
        self.target = None

    def check(self, action, observation, remaining, step, before=None):
        take = take_target(action)
        if take and task_mentions(self.task, take[0]):
            self.target = take[0]
            expected = {'predicate': 'holding', 'object': take[0], 'source_receptacle': take[1]}
            match = PICKUP.fullmatch(norm(observation))
            status = 'unknown' if not match else ('confirmed' if match.groups() == take else 'contradicted')
            holding_status = 'unknown' if not match else ('confirmed' if match[1] == take[0] else 'contradicted')
            reason = ('exact pickup object and source match' if status == 'confirmed' else
                      'pickup feedback names a different object or source' if status == 'contradicted' else
                      'no supported exact pickup confirmation; possession cannot be inferred')
        elif norm(action) == 'inventory' and self.target:
            expected = {'predicate': 'holding', 'object': self.target}
            objects = inventory_objects(observation)
            status = 'unknown' if objects is None else ('confirmed' if self.target in objects else 'contradicted')
            holding_status = status
            reason = 'current inventory explicitly lists target' if status == 'confirmed' else (
                'complete current inventory excludes target' if status == 'contradicted' else 'unrecognized inventory feedback')
        else:
            return {'applicable': False, 'status': 'unknown', 'reason': 'outside acquire/verify-possession scope',
                    'blocks_tail': False, 'dependent_actions': []}
        affected = dependencies(expected['object'], remaining)
        before = before or {}
        return {'applicable': True, 'status': status, 'expected': expected, 'reason': reason,
                'precondition': {'description': 'at source; target accessible; pickup allowed',
                    'admissible_before': (action.lower() in {a.lower() for a in before['admissible']}) if 'admissible' in before else None,
                    'source': 'public admissible commands immediately before this action',
                    'observation_before': before.get('observation'), 'step': step},
                'evidence': {'source': 'env.step feedback', 'action': action, 'step': step,
                    'observed_at_step': step, 'age_steps': 0, 'scope': 'after this action only',
                    'text': observation, 'historical_state_reused': False},
                'dependent_actions': affected, 'holding_evidence_status':holding_status,
                'blocks_tail': bool(affected and holding_status != 'confirmed'),
                'suggested_observation': 'inventory' if status == 'unknown' else None,
                'task_completed': None}

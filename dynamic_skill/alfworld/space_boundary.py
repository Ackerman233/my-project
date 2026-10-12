"""SPACE-inspired inference scaffolding, NOT SPACE training or distilled policy."""
import re

GUIDANCE = '''
Chunk-boundary policy (inference scaffold, not a trained SPACE policy):
Choose one local routine and emit its consecutive primitive actions, up to the stated limit.
Prefer 2-4 actions when a routine is determined; one action is correct at a genuine information boundary.
Do not stop merely because the next command is absent from Admissible now: that list describes ONLY the current state.
Predictable transitions can be composed, e.g. a currently admissible take, go to a known destination, and move the same object there if its receptacle is known accessible.
For an unknown container, visit/open then STOP to read its contents before choosing an object ID.
For known visible/held objects, acquisition, navigation, and delivery can form one routine; do not re-observe after every deterministic transition.
For search, several known unsearched locations may form a short scan, with no guessed take actions; do not revisit already inspected locations without new evidence.
Examples are schematic, not facts about this episode:
  Visible egg 7 on table 2, known open target surface counter 3: ["take egg 7 from table 2", "go to counter 3", "move egg 7 to counter 3"].
  Known closed drawer 2: ["go to drawer 2", "open drawer 2"], then read the resulting contents.
Never invent object IDs, pad with inventory/no-ops, force a minimum length, or cross a boundary requiring a newly discovered ID.
In <thought>, briefly name the local routine and why this chunk ends; keep the same thought/actions format.
After an error, repair the specific violated precondition, not a generic ritual. Resume batching when justified.
A target-discovery notification is progress, NOT a failed action: consider the newly available exact take action before more searching.
'''

def target_type(task):
    m=re.match(r'(?:put|examine) (?:some|a|an|the) ([a-z]+)\b',task.lower().strip())
    return m[1] if m else None  # restrict to simple task grammar, no guessed semantic parsing

def discovery_check(task, action, remaining, admissible_after, step):
    result={'triggered':False,'kind':'target_discovery','step':step,
            'source':'public admissible commands after current env.step','age_steps':0}
    target=target_type(task)
    if not target or not remaining or not action.startswith(('go to ','open ','examine ')):
        return result
    # Only truncate an exploration-only tail; never interrupt queued acquisition/delivery.
    if not all(a.startswith(('go to ','open ','examine ')) for a in remaining):return result
    for command in admissible_after:
        m=re.fullmatch(r'take ('+re.escape(target)+r' [0-9]+) from ([a-z][a-z0-9]* [0-9]+)',command)
        if m and remaining[0].startswith('go to ') and remaining[0][6:]!=m[2]:
            result.update(triggered=True,object=m[1],source_receptacle=m[2],
                evidence_command=command,reason='target discovered and currently takeable; queued search would leave it behind')
            return result
    return result

def boundary_prompt(body):
    return body.replace('Stop before unknown information.',
                        'Stop before choosing actions whose required object IDs or branching outcomes are still unknown.')+GUIDANCE

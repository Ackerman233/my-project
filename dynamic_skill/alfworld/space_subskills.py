"""Hand-authored parameterized subskills; no inferred solutions or auto planning."""
import re
from dynamic_skill.alfworld.micro_loop_protocol import SYSTEM
SYSTEM_SKILL=SYSTEM.replace('concrete command','primitive command or @inspect/@scan subskill').replace(
    'Do not output native thinking tags, programs, or markdown.',
    'The user prompt defines allowed @inspect/@scan subskills. No other code, native thinking tags, or markdown.')
INSTRUCTIONS='''
Skill-augmented interface (overrides primitive-only actions format):
Prefer a suitable local subskill over separate LLM calls for its commands.
Return the SAME <thought>brief decision</thought><actions>[...]</actions> format.
Array items may be primitive actions or subskills. Their expanded commands execute in array order; total expanded length must fit the limit.
  "@inspect drawer 2" expands to ["go to drawer 2", "open drawer 2"]. Valid for known drawer/cabinet/fridge/microwave IDs.
  If already at the location, @inspect omits navigation: it opens a currently closed container, or examines an already accessible container/surface. It uses only the current public admissible commands to recognize this case.
  Use @inspect for an uninspected container; if already known open, use ordinary navigation. Runtime rejection ends the queue; no guessed take action follows open.
  "@inspect shelf 1" is a one-action surface visit (no opening). Never open a surface.
  ["@scan countertop 1", "@scan countertop 2"] expands to ["go to countertop 1", "go to countertop 2"]. A single @scan item can also use | between known surfaces.
  Scanning is for finding objects, not transporting them. Prefer unsearched surfaces; never insert unknown IDs or inspect already searched containers without reason.
All expanded actions count toward the SAME chunk limit and environment budget.
Count primitive actions, not array items: a remote container @inspect costs TWO actions; a surface visit or current-location inspect costs ONE; @scan costs one per location.
For a limit of 4, ["@inspect drawer 1", "@inspect drawer 2"] costs 4 and fits. Three remote container inspections cost 6 and are forbidden. Split longer searches across later decisions. If unsure of the total, choose ONE local subskill.
Use known IDs only. Single primitive actions remain allowed when necessary.
When the target is visible/takeable, acquire it instead of scanning further. A discovery interruption means stop searching and use the new exact object evidence.
'''
CONTAINERS={'drawer','cabinet','fridge','microwave'}
SURFACES={'countertop','diningtable','sidetable','coffeetable','desk','shelf','dresser','bed','sofa','armchair','stoveburner','toilet','bathtubbasin','sinkbasin','ottoman'}

def normalize_response(text):
    """Only repair omitted actions tags around an otherwise exact JSON array.

    No action guessing, execution, model retry, or acceptance of extra prose.
    Caller still validates JSON, length, arguments and expanded preconditions.
    """
    m=re.fullmatch(r'\s*(<thought>[^<]+</thought>)\s*(\[.*\])\s*',text,re.S)
    if not m:return text,False
    return m[1]+'<actions>'+m[2]+'</actions>',True

def expand(actions,context,limit):
    if not any(a.startswith('@') for a in actions):return actions,None
    known=set(re.findall(r'\b[a-z][a-z0-9]* [0-9]+\b',context['initial']))
    known.update(a[6:] for a in context['admissible'] if a.startswith('go to '))
    expanded=[];skills=[]
    for raw in actions:
        if not raw.startswith('@'):
            expanded.append(raw);continue
        parts=raw.split(' ',1)
        if len(parts)!=2:raise ValueError('subskill syntax')
        name,args=parts
        if name=='@inspect':
            kind=args.rsplit(' ',1)[0]
            if args not in known or kind not in CONTAINERS|SURFACES:raise ValueError('unknown inspect location')
            if not expanded and 'go to '+args not in context['admissible'] and 'open '+args in context['admissible']:
                expanded+=['open '+args]
            elif not expanded and 'go to '+args not in context['admissible'] and 'examine '+args in context['admissible']:
                expanded+=['examine '+args]
            else:
                expanded+=['go to '+args]+(['open '+args] if kind in CONTAINERS else [])
        elif name=='@scan':
            locations=args.split('|')
            if not 1<=len(locations)<=limit or len(set(locations))!=len(locations):raise ValueError('scan bounds')
            if any(x not in known or x.rsplit(' ',1)[0] not in SURFACES for x in locations):raise ValueError('unknown surface')
            expanded+=['go to '+x for x in locations]
        else:raise ValueError('unknown skill')
        skills.append({'skill':name,'arguments':args})
    if len(expanded)>limit:raise ValueError(f'expanded action budget exceeded: planned={len(expanded)} limit={limit}')
    if expanded[0] not in context['admissible']:raise ValueError('subskill precondition')
    return expanded,{'skills':skills,'model_actions':actions,'expanded_actions':expanded,
                     'origin':'hand_authored_parameterized_subskill','not_model_emitted_primitives':True}

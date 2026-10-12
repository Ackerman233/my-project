"""Public-trajectory search/state audit; no hidden environment state or API access."""
import argparse
import re
from collections import Counter
from pathlib import Path
from experiments.interrupt_recovery_ablation import read, digest
from experiments.puzzle_pilot import save

ENTITY = r'[a-z][a-z0-9]* [0-9]+'


def target(task):
    for pattern in [r'put (?:some|a|an|the) ([a-z]+)\b',
                    r'examine (?:some|a|an|the) ([a-z]+)\b', r'look at ([a-z]+)\b']:
        match = re.match(pattern, task.lower())
        if match:
            return match[1]


def analyze_episode(path):
    initial = read(path.parent / 'initial.json')
    trace = read(path)
    result = read(path.parent / 'result.json')
    obj = target(initial['task'])
    known, searched, visited = set(), {}, Counter()
    events, opportunities = [], []
    holding = None
    for i, step in enumerate(trace):
        before = step['admissible_before']
        known.update(a[6:] for a in before if a.startswith('go to '))
        feedback = step['env_feedback']
        action = step['action']
        if action.startswith('go to '):
            location = action[6:]
            visited[location] += 1
            if visited[location] > 1:
                events.append({'step': i, 'kind': 'revisit', 'location': location,
                    'previous_content_observation': searched.get(location),
                    'note': 'A revisit is not automatically unnecessary; transport/lighting may require it.'})
        content = re.search(r'(?:On the|In the) (' + ENTITY + r'), you see (.+)', feedback)
        opened = re.match(r'You open the (' + ENTITY + r')\..*In it, you see (.+)', feedback)
        for match in [content, opened]:
            if match:
                searched[match[1]] = {'step': i, 'feedback': feedback}
        pickup = re.fullmatch(r'You pick up the (' + ENTITY + r') from the (' + ENTITY + r')\.', feedback)
        moved = re.fullmatch(r'You move the (' + ENTITY + r') to the (' + ENTITY + r')\.', feedback)
        if pickup:
            holding = pickup[1]
            events.append({'step': i, 'kind': 'confirmed_pickup', 'object': holding})
        if moved and holding == moved[1]:
            holding = None
        after = trace[i + 1]['admissible_before'] if i + 1 < len(trace) else None
        tail = step.get('remaining_before_interrupt', step.get('remaining_actions', []))
        takeable = [a for a in (after or []) if obj and re.fullmatch(
            r'take ' + re.escape(obj) + r' [0-9]+ from ' + ENTITY, a)]
        if takeable:
            events.append({'step': i, 'kind': 'target_takeable', 'commands': takeable,
                           'tail': tail, 'has_nonempty_tail': bool(tail)})
        surface = re.search(r'On the (' + ENTITY + r'), you see ', feedback)
        if tail and after is not None and tail[0].startswith('open ') and tail[0] not in after and surface:
            opportunities.append({'step': i, 'kind': 'surface_open_contradiction', 'tail': tail,
                'evidence': {'feedback': feedback, 'admissible_after': after},
                'reason': 'queued open has no public admissible support and latest feedback describes a surface'})
        elif tail and feedback.strip().lower() == 'nothing happens.':
            opportunities.append({'step': i, 'kind': 'rejected_action_nonempty_tail', 'tail': tail,
                'evidence': {'feedback': feedback, 'admissible_after': after},
                'reason': 'real rejected action with an unexecuted model-generated tail'})
        elif tail and takeable and tail[0].startswith('go to '):
            source = takeable[0].rsplit(' from ', 1)[1]
            if tail[0][6:] != source and all(a.startswith(('go to ', 'open ', 'examine ')) for a in tail):
                opportunities.append({'step': i, 'kind': 'target_discovery_before_departure', 'tail': tail,
                    'evidence': {'feedback': feedback, 'admissible_after': after, 'take_commands': takeable},
                    'reason': 'currently takeable target with a search-only tail leaving its location'})
    return {'episode': str(path.parent), 'source_trace': str(path), 'source_sha256': digest(path),
        'task': initial['task'], 'task_id': result['task_id'], 'status': result['status'],
        'target_type': obj, 'known_public_locations': sorted(known),
        'content_observed_locations': sorted(searched),
        'content_not_observed_locations': sorted(known - set(searched)),
        'visit_counts': dict(visited), 'revisit_count': sum(max(0, n - 1) for n in visited.values()),
        'last_confirmed_held_object': holding, 'events': events, 'opportunities': opportunities,
        'scope': 'content observations are historical; location count is public search coverage, not hidden object completeness'}


def audit(roots, output):
    episodes = []
    for root in roots:
        for path in sorted(root.glob('online/*/predictions/*/conversation.json')):
            if not read(path.parent / 'result.json')['success']:
                episodes.append(analyze_episode(path))
    cases = []
    # Earliest state per episode/kind; expose both redundant-open and real rejection cases.
    for episode in episodes:
        seen = set()
        for opportunity in episode['opportunities']:
            if opportunity['kind'] in seen:
                continue
            seen.add(opportunity['kind'])
            cases.append({'id': f'prefix-{len(cases)}', 'source_trace': episode['source_trace'],
                'source_sha256': episode['source_sha256'], 'task_id': episode['task_id'],
                **opportunity})
    report = {'paid_calls': 0, 'failed_episodes': len(episodes), 'episodes': episodes,
              'cases': cases, 'selection': 'all failed final-version episodes; earliest real-tail opportunity per episode/kind, no injected plans',
              'note': 'Final-step public state must be re-established by real prefix replay before any treatment.'}
    save(output, report)
    print({'failed_episodes': len(episodes), 'cases': len(cases),
           'opportunity_kinds': dict(Counter(c['kind'] for c in cases))})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, default=[
        Path('results/protocol_diagnostic_v5_20261007'), Path('results/subskill_exposure_20261007'),
        Path('results/subskill_fixed_20261007')])
    parser.add_argument('--output', type=Path, default=Path('results/prefix_recovery_20261007/search_audit.json'))
    args = parser.parse_args()
    audit(args.roots, args.output)

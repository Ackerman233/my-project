"""Opt-in ALFWorld adapter: review a queued open after fresh surface feedback.

No model calls, no hidden state, no default-runner changes. None means outside
this adapter's narrow scope; it is not evidence that an arbitrary queue is safe.
"""
import re
from dynamic_skill.evidence.queue_evidence import AlfworldEvidence, ENTITY
from dynamic_skill.evidence.queue_revision import revise_head


def review_open_head(actions, observation, admissible, *, observed_id=None):
    if not actions:
        return None
    opened = re.fullmatch(r'open (' + ENTITY + ')', actions[0])
    if not opened or not re.search(r'On the ' + re.escape(opened[1]) + r', you see ', observation):
        return None
    adapter = AlfworldEvidence(observation, admissible, complete=False, observed_id=observed_id)
    return revise_head(actions, adapter)

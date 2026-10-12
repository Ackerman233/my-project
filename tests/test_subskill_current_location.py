"""Regressions for an observed real-model inspect failure at a closed drawer."""
import pytest
from dynamic_skill.space_subskills import expand


@pytest.mark.parametrize('admissible,expected', [
    (['open drawer 2', 'examine drawer 2'], ['open drawer 2']),
    (['close drawer 2', 'examine drawer 2'], ['examine drawer 2']),
    (['examine desk 1'], ['examine desk 1']),
])
def test_inspect_current_location_uses_public_commands(admissible, expected):
    location = expected[0].split(' ', 1)[1]
    actions, provenance = expand(['@inspect ' + location],
        {'initial': 'drawer 2, desk 1', 'admissible': admissible}, 1)
    assert actions == expected
    assert provenance['model_actions'] == ['@inspect ' + location]


def test_current_location_cannot_be_inferred_from_missing_navigation_alone():
    with pytest.raises(ValueError, match='precondition'):
        expand(['@inspect drawer 2'], {'initial': 'drawer 2', 'admissible': ['look']}, 4)


def test_remote_closed_container_keeps_two_action_inspection():
    assert expand(['@inspect drawer 2'],
        {'initial': 'drawer 2', 'admissible': ['go to drawer 2']}, 4)[0] == ['go to drawer 2', 'open drawer 2']


def test_current_evidence_not_reused_after_navigation_in_same_queue():
    actions, _ = expand(['go to desk 1', '@inspect drawer 2'],
        {'initial': 'drawer 2, desk 1', 'admissible': ['go to desk 1', 'open drawer 2']}, 4)
    assert actions == ['go to desk 1', 'go to drawer 2', 'open drawer 2']

from dynamic_skill.checker import CheckStatus, ObservationChecker
from dynamic_skill.interrupter import DecisionType
from dynamic_skill.skill_executor import ChunkExecutor


def test_checker_interrupts_invalid_action() -> None:
    result = ObservationChecker().check("take apple", "That is not a valid action.")
    assert result.status is CheckStatus.INVALID_ACTION
    assert result.should_interrupt


def test_checker_detects_three_action_loop() -> None:
    result = ObservationChecker().check("look", "same room", ["look", "look"])
    assert result.status is CheckStatus.ACTION_LOOP


def test_dynamic_executor_replans_on_rejection() -> None:
    executor = ChunkExecutor(chunk_size=4, dynamic=True)
    executor.start("Your task is to: put an apple on the table")
    event = executor.observe("take apple", "That is not a valid action.")
    assert event.decision.kind is DecisionType.REPLAN
    assert event.interruption_count == 1
    assert event.plan_revision == 1


def test_fixed_executor_waits_until_chunk_boundary() -> None:
    executor = ChunkExecutor(chunk_size=2, dynamic=False)
    executor.start("Your task is to: put an apple on the table")
    first = executor.observe("look", "No change.")
    second = executor.observe("look", "No change.")
    assert first.decision.kind is DecisionType.CONTINUE
    assert second.decision.kind is DecisionType.REPLAN


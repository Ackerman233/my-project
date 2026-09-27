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
    executor.load_chunk(["take apple", "put apple on table"])
    event = executor.observe(executor.next_action(), "That is not a valid action.")
    assert event.decision.kind is DecisionType.REPLAN
    assert event.interruption_count == 1
    assert event.plan_revision == 1
    assert event.discarded_actions == ("put apple on table",)
    assert executor.needs_plan


def test_fixed_executor_waits_until_chunk_boundary() -> None:
    executor = ChunkExecutor(chunk_size=2, dynamic=False)
    executor.start("Your task is to: put an apple on the table")
    executor.load_chunk(["look", "look"])
    first = executor.observe(executor.next_action(), "Nothing happens.")
    second = executor.observe(executor.next_action(), "Nothing happens.")
    assert first.decision.kind is DecisionType.CONTINUE
    assert second.decision.kind is DecisionType.ADVANCE
    assert executor.interruption_count == 0
    assert executor.boundary_count == 1


def test_checker_distinguishes_abab_from_abbb():
    checker = ObservationChecker()
    assert checker.check("b", "room", ["a", "b", "a"]).status is CheckStatus.ACTION_LOOP
    assert checker.check("a", "room", ["a", "b", "a"]).status is not CheckStatus.ACTION_LOOP


def test_last_action_anomaly_is_not_a_mid_chunk_interrupt():
    executor = ChunkExecutor()
    executor.start("task")
    executor.load_chunk(["bad"])
    event = executor.observe(executor.next_action(), "Nothing happens.")
    assert event.decision.kind is DecisionType.REPLAN
    assert executor.interruption_count == 0
    assert executor.anomaly_replan_count == 1

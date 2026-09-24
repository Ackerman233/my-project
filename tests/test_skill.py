from dynamic_skill.skill_induction import SkillInducer


def test_induction_adds_transform_for_clean_task() -> None:
    plan = SkillInducer().induce("Your task is to: clean an apple and put it on a table")
    assert any(step.step_id == "transform" for step in plan.steps)
    assert plan.task.startswith("clean an apple")


def test_replan_is_revisioned_and_preserves_remaining_work() -> None:
    inducer = SkillInducer()
    plan = inducer.induce("Your task is to: put an apple on a table")
    revised = inducer.replan(plan, 1, "stalled", "same room")
    assert revised.revision == 1
    assert revised.steps[0].step_id == "recover_1"
    assert revised.steps[1:] == plan.steps[1:]


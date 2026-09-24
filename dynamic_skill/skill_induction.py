from __future__ import annotations

import re

from .skill import SkillPlan, SkillStep


class SkillInducer:
    """Deterministic ALFWorld plan induction; intentionally training-free."""

    def __init__(self, chunk_size: int = 4) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    @staticmethod
    def extract_task(observation: str) -> str:
        match = re.search(r"Your task is to:\s*(.+)", observation, re.I | re.S)
        if match:
            return match.group(1).strip().splitlines()[0]
        return observation.strip().splitlines()[0][:300] if observation.strip() else "unknown task"

    def _step(self, step_id: str, objective: str, *signals: str) -> SkillStep:
        return SkillStep(step_id, objective, tuple(signals), self.chunk_size)

    def induce(self, observation: str, task: str | None = None) -> SkillPlan:
        task = (task or self.extract_task(observation)).strip()
        lower = task.lower()
        steps = [self._step("locate", "Systematically locate the required object", "you see", "you open")]
        steps.append(self._step("acquire", "Take the required object when visible", "you pick up", "you take"))

        if "clean" in lower or "wash" in lower:
            steps.append(self._step("transform", "Clean the held object at a sink basin", "you clean", "cleaned"))
        elif "heat" in lower or "hot" in lower:
            steps.append(self._step("transform", "Heat the held object with the microwave", "you heat", "heated"))
        elif "cool" in lower or "cold" in lower:
            steps.append(self._step("transform", "Cool the held object in the fridge", "you cool", "cooled"))
        elif "look at" in lower or "examine" in lower:
            steps.append(self._step("inspect", "Carry the object to a lamp and examine it under light", "you turn on", "examined"))

        if "two" in lower:
            steps.extend(
                [
                    self._step("place_first", "Place the first object at the destination", "you put", "you place"),
                    self._step("acquire_second", "Locate and take the second matching object", "you pick up", "you take"),
                ]
            )
        steps.append(self._step("deliver", "Navigate to the destination and place the object", "you put", "you place"))
        steps.append(self._step("verify", "Verify that all goal conditions are satisfied", "task completed", "success"))
        return SkillPlan("alfworld-training-free", task, tuple(steps))

    def replan(
        self,
        plan: SkillPlan,
        active_index: int,
        reason: str,
        observation: str,
    ) -> SkillPlan:
        remaining = plan.steps[max(0, active_index):]
        recovery = self._step(
            f"recover_{plan.revision + 1}",
            "Recover from the detected failure: inspect admissible actions, choose a different valid action, and avoid repeating the failed transition",
            "you see",
            "you open",
            "you move",
        )
        metadata = dict(plan.metadata)
        metadata.update({"last_interrupt": reason, "last_observation": observation[-500:]})
        return SkillPlan(
            name=plan.name,
            task=plan.task,
            steps=(recovery, *remaining),
            revision=plan.revision + 1,
            metadata=metadata,
        )


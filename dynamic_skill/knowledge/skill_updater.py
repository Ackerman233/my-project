from __future__ import annotations

from dataclasses import asdict

from dynamic_skill.knowledge.failure_analyzer import FailureReport


class SkillUpdater:
    """Produces inspectable suggestions; it never mutates the official skill."""

    def suggest(self, report: FailureReport) -> dict:
        return {
            "operation": "append_rule",
            "section": "Dynamic recovery rules",
            "rule": report.recommendation,
            "source": asdict(report),
            "requires_review": True,
        }

    def apply_to_text(self, skill_text: str, suggestion: dict) -> str:
        rule = str(suggestion["rule"]).strip()
        return f"{skill_text.rstrip()}\n\n## Dynamic recovery rules\n\n- {rule}\n"


"""Disclose domain tools only while this turn's skill reads are in context."""

from deepagents.middleware.skills import SkillsMiddleware


class TurnSkillsMiddleware(SkillsMiddleware):
    @property
    def name(self) -> str:
        # Replace the built-in SkillsMiddleware in its original graph slot.
        return "SkillsMiddleware"

    def _disclose_skill_tools(self, request, reads, resolved):
        # Skill access ends at the next user message. The harness may place
        # pinned skill instructions immediately before the current message.
        current_user = next(
            (
                index for index in range(len(request.messages) - 1, -1, -1)
                if request.messages[index].type == "human"
                and not request.messages[index].additional_kwargs.get("lc_source")
            ),
            0,
        )
        while current_user and request.messages[current_user - 1].additional_kwargs.get("lc_source") == "pinned_skill":
            current_user -= 1
        current_reads = [read for read in reads if read[0] >= current_user]
        return super()._disclose_skill_tools(request, current_reads, resolved)

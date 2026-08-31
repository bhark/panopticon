"""The knowledge base and its jury."""

from __future__ import annotations

from panopticon.model import Submission, Truth, VerdictCall


class Knowledge:
    NEEDED_TRUE = 2

    def __init__(self) -> None:
        self.truths: list[Truth] = []
        self.pending: dict[str, Submission] = {}

    def render(self) -> str: ...
    def render_pending(self) -> str: ...
    def submit(self, agent: str, title: str, body: str) -> Submission: ...
    def join(self, agent: str, submission_id: str) -> Submission:
        """Raises ValueError if the agent submitted it or is already seated."""
        ...

    def leave(self, agent: str, submission_id: str) -> Submission: ...
    def verdict(
        self, agent: str, submission_id: str, call: VerdictCall, reasoning: str,
        restated_title: str = "", restated_body: str = "",
    ) -> tuple[Submission, str]:
        """Returns (submission, outcome) where outcome is accepted|rejected|restated|pending."""
        ...

"""The knowledge base and its jury."""

from __future__ import annotations

from panopticon import store
from panopticon.model import Submission, Truth, Verdict, VerdictCall


class Knowledge:
    NEEDED_TRUE = 2

    def __init__(self) -> None:
        self.truths: list[Truth] = []
        self.pending: dict[str, Submission] = {}
        self._submitted = 0

    def render(self) -> str:
        if not self.truths:
            return "knowledge base: empty"
        lines = [f"knowledge base ({len(self.truths)} truths)"]
        for truth in self.truths:
            lines.append(f"[{truth.id}] {truth.title}")
            lines.append(f"  {truth.body}")
        return "\n".join(lines)

    def render_pending(self) -> str:
        if not self.pending:
            return "jury queue: empty"
        lines = [f"jury queue ({len(self.pending)} waiting)"]
        for sub in self.pending.values():
            trues = sum(1 for v in sub.verdicts if v.call is VerdictCall.TRUE)
            jurors = ", ".join(sub.jurors) or "none"
            lines.append(
                f"[{sub.id}] {sub.title} | by {sub.submitted_by} | jurors: {jurors} "
                f"| {trues}/{self.NEEDED_TRUE} true"
            )
            lines.append(f"  {sub.body}")
        return "\n".join(lines)

    def submit(self, agent: str, title: str, body: str) -> Submission:
        self._submitted += 1
        sub = Submission(
            id=f"s{self._submitted}", title=title, body=body, submitted_by=agent, barred=[agent]
        )
        self.pending[sub.id] = sub
        return sub

    def join(self, agent: str, submission_id: str) -> Submission:
        """Raises ValueError if the agent submitted it or is already seated."""
        sub = self._waiting(submission_id)
        if agent in sub.barred:
            raise ValueError(f"{sub.id} is your own statement, you cannot judge it")
        if agent in sub.jurors:
            raise ValueError(f"you are already on the jury for {sub.id}")
        sub.jurors.append(agent)
        return sub

    def leave(self, agent: str, submission_id: str) -> Submission:
        sub = self._waiting(submission_id)
        if agent not in sub.jurors:
            raise ValueError(f"you are not on the jury for {sub.id}")
        sub.jurors.remove(agent)
        return sub

    def verdict(
        self,
        agent: str,
        submission_id: str,
        call: VerdictCall,
        reasoning: str,
        restated_title: str = "",
        restated_body: str = "",
    ) -> tuple[Submission, str]:
        """Returns (submission, outcome) where outcome is accepted|rejected|restated|pending."""
        sub = self._waiting(submission_id)
        if agent in sub.barred:
            raise ValueError(f"{sub.id} is your own statement, you cannot judge it")
        if agent not in sub.jurors:
            raise ValueError(f"you are not on the jury for {sub.id}")
        if any(v.juror == agent for v in sub.verdicts):
            raise ValueError(f"you have already ruled on {sub.id}")
        if call is VerdictCall.RESTATE and not (restated_title and restated_body):
            raise ValueError("a restate verdict must carry the replacement title and body")

        sub.verdicts.append(Verdict(juror=agent, call=call, reasoning=reasoning))

        if call is VerdictCall.RESTATE:
            return self._restate(agent, sub, restated_title, restated_body), "restated"

        if call is VerdictCall.FALSE:
            self._drop(sub)
            return sub, "rejected"

        if sum(1 for v in sub.verdicts if v.call is VerdictCall.TRUE) >= self.NEEDED_TRUE:
            self._drop(sub)
            self.truths.append(
                Truth(
                    id=f"k{len(self.truths) + 1}",
                    title=sub.title,
                    body=sub.body,
                    submitted_by=sub.submitted_by,
                )
            )
            return sub, "accepted"

        return sub, "pending"

    # persistence

    def snapshot(self) -> dict:
        return {
            "truths": [store.dump(t) for t in self.truths],
            "pending": [store.dump(s) for s in self.pending.values()],
            # ids must not restart after a resume, or a new submission reuses a dropped id
            "submitted": self._submitted,
        }

    def restore(self, raw: dict) -> None:
        self.truths = [store.load(Truth, t) for t in raw.get("truths", [])]
        self.pending = {s.id: s for s in map(store.restore_submission, raw.get("pending", []))}
        self._submitted = raw.get("submitted", len(self.pending))

    # internals

    def _restate(self, agent: str, sub: Submission, title: str, body: str) -> Submission:
        self._submitted += 1
        replacement = Submission(
            id=f"s{self._submitted}",
            title=title,
            body=body,
            submitted_by=agent,
            jurors=[j for j in sub.jurors if j != agent],
            restated_from=sub.id,
            barred=[*sub.barred, agent],
        )
        self._drop(sub)
        self.pending[replacement.id] = replacement
        return replacement

    def _drop(self, sub: Submission) -> None:
        self.pending.pop(sub.id, None)

    def _waiting(self, submission_id: str) -> Submission:
        sub = self.pending.get(submission_id)
        if sub is None:
            raise ValueError(f"no submission {submission_id} is waiting for a jury")
        return sub

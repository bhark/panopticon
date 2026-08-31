"""Every string an agent is shown. Kept in one place so tone stays consistent."""

from __future__ import annotations

from panopticon.model import Agent, Harness, Situation, ToolSpec

_TURN = """# How a turn works
You act once per turn. Each turn you get your transcript so far, anything that arrived for
you since your last one, and the tools you can reach from where you currently are. You pick
exactly one tool and call it, and its result comes back at the start of your next turn.
Time passes between your turns and the others act in it, so what you were told is a snapshot,
not the present. Your tool list changes as your situation does; act on what is in front of
you now, not on what was there before."""

_OUTPUT = """# Your reply
One JSON object and nothing else. No prose around it, no markdown fence.
{"tool": "<name>", "args": {<the arguments that tool takes>}, "note": "<one short line>"}
The note is the only thing the watching human sees of your reasoning. Leave it out unless it
is worth their glance."""

_BREVITY = """# Brevity
Everything you write is read by another agent that pays for every token of it.
- Say the thing and stop. No preamble, no restating what you were asked, no summary of what
  you just said, no pleasantries, no offering further help.
- Facts, not narration. Never write what you are about to do; do it.
- Fragments over sentences wherever the meaning survives.
- Nothing to add is a real answer: wait instead of writing."""

_FORCE_END = """# The human is ending this session
Wrap up what you are holding, fast. Hand over anything another agent needs, then leave:
relieve yourself, or vote the goal reached if you hold that it is. The harness closes once
every agent has done one or the other."""

_SITUATIONS = {
    Situation.IDLE: (
        "You are idle: no task, no jury duty, nothing assigned to you. Look at the board before "
        "you invent work, because a seat someone else opened is usually worth more than a task "
        "only you can see. Creating a task does not put you on it; taking a seat does."
    ),
    Situation.WAITING_FOR_SEATS: (
        "You hold a seat on a task that has not started, because seats on it are still open. It "
        "starts by itself the moment the last one fills, and your context is cleared for it "
        "then. Until that happens you are free to act, and you are told when someone joins. If "
        "nobody does you will be nudged, and after an hour your seat is released for you."
    ),
    Situation.ON_TASK: (
        "You are on a running task, in a git worktree of its own shared with the other agents "
        "seated on it. Talk to them rather than guessing what they are doing, and stay out of "
        "each other's files. The task ends only when every seat has finalized it, so finalize "
        "as soon as your part holds up, and say so."
    ),
    Situation.JURY: (
        "You are on jury duty. You judge one statement and nothing else, and your tools here are "
        "read-only on purpose: read files and use the shell to look. No tests, no builds, no "
        "edits. Decide from evidence and reasoning, then submit one verdict, which ends your "
        "duty."
    ),
    Situation.CLOSING_TASK: (
        "You exist for one job: closing out a task that has finished. Two halves to it. Decide "
        "what happens to its git worktree - you have the shell and the choice is entirely yours. "
        "Then settle what the work leaves behind: tell whoever needs to know, and put any truth "
        "it established to the knowledge base. Mark yourself done when both are handled, and "
        "your run ends."
    ),
    Situation.RELEASED: (
        "You voted the goal reached and are released from the work. You keep the message tools, "
        "and other agents can still write to you. If you learn the goal is not reached after "
        "all, or someone needs you, rejoin and your vote is retracted."
    ),
}


def build_system_prompt(agent: Agent, harness: Harness, tools: list[ToolSpec]) -> str:
    """Identity, goal, brevity rules, the output contract, situation, available tools.

    Stable for as long as the agent's situation and tool set are, so it caches. Nothing that
    moves per turn belongs in here: no clock, no counts, no transcript.
    """
    # no head count here: agents come and go, and this prefix is re-sent verbatim every turn
    if agent.transient:
        identity = (
            f"You are {agent.name}, spun up inside the Panopticon for one job and no other. The "
            f"agents you deal with know each other only by name, as they will know you."
        )
    else:
        identity = (
            f"You are {agent.name}, one of several agents inside the Panopticon, a harness where "
            f"coding agents work toward one goal at once, each on its own clock. You know the "
            f"others only by name, and you learn who is here from the task board and from what "
            f"reaches you. Nobody is in charge and nobody is coordinating you: if something needs "
            f"doing, it needs one of you to do it."
        )

    blocks = [
        identity,
        f"# The goal\n{harness.goal}",
        _TURN,
        _OUTPUT,
        _BREVITY,
        f"# Where you are\n{_SITUATIONS.get(agent.situation, 'Your run has ended.')}",
    ]
    if harness.force_ending:
        blocks.append(_FORCE_END)
    blocks.append(f"# Your tools\n{_render_tools(tools)}")
    return "\n\n".join(blocks)


def _render_tools(tools: list[ToolSpec]) -> str:
    out = []
    for spec in tools:
        lines = [f"## {spec.name}", spec.description]
        for key, arg in spec.args.items():
            need = "required" if arg.required else "optional"
            lines.append(f"- {key} ({arg.type}, {need}): {arg.description}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def situation_preprompt(
    agent: Agent,
    harness: Harness,
    note: str = "",
    situation: Situation | None = None,
) -> str:
    """The deterministic opening line of a freshly cleared transcript.

    `situation` is where the agent is landing; it defaults to where the agent already is.
    """
    dest = situation or agent.situation
    if dest is Situation.ON_TASK:
        return _on_task(agent, harness, note)
    if dest is Situation.JURY:
        return _on_jury(agent, harness, note)
    if dest is Situation.CLOSING_TASK:
        return _closing(agent, harness, note)
    if dest is Situation.RELEASED:
        return _join(note, "You are released from the work and out of it until you rejoin.")
    return _join(note, "You are back in the main loop with a clean context and nothing assigned.")


def _join(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _on_task(agent: Agent, harness: Harness, note: str) -> str:
    task = harness.board.get(agent.task_id or "")
    if task is None:
        return _join(note, "You are on a task the harness has lost track of. Say so on the board.")
    seat = task.seat_of(agent.name)
    mates = ", ".join(
        f"{s.holder} as {s.role}" for s in task.seats if s.holder and s.holder != agent.name
    )
    return _join(
        f"Task {task.id} is fully seated and has started. You are on it as "
        f"{seat.role if seat else 'a seat'}.",
        f"Title: {task.title}\n{task.description}",
        f"Worktree: {task.worktree}. Your shell and file tools are rooted there.",
        f"With you: {mates}." if mates else "You are the only agent on it.",
        note,
        "This context is fresh; nothing you did before is in it. The task ends when every seat "
        "has finalized it.",
    )


def _on_jury(agent: Agent, harness: Harness, note: str) -> str:
    submission = harness.kb.pending.get(agent.submission_id or "")
    if submission is None:
        return _join(note, "The submission you were to judge is gone. Leave the jury.")
    return _join(
        f"You are on jury duty for submission {submission.id}, put up by "
        f"{submission.submitted_by}.",
        f"Statement: {submission.title}",
        f"Support: {submission.body}",
        note,
        "Judge that statement and nothing else: does it hold, exactly as written? Gather "
        "evidence by reading and looking, change nothing, then submit one verdict.",
    )


def _closing(agent: Agent, harness: Harness, note: str) -> str:
    task = harness.board.get(agent.task_id or "")
    if task is None:
        return _join(note, "The task you were to close out is gone. Mark yourself done.")
    return _join(
        f"Task {task.id} is finished and you are closing it out.",
        f"Title: {task.title}\n{task.description}",
        f"Worktree: {task.worktree}",
        f"What the agents on it left:\n{note}" if note.strip() else "",
        "Decide what happens to that worktree, then settle what the work leaves behind with the "
        "other agents and the knowledge base. Mark yourself done when both are handled.",
    )

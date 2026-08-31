"""Which tools an agent can reach, given where it currently is."""

from __future__ import annotations

from panopticon.model import Agent, Harness, Situation, ToolSpec

WAIT = "wait"
VIEW_BOARD = "view_task_board"
CREATE_TASK = "create_task"
ASSIGN_SELF = "assign_self"
UNASSIGN_SELF = "unassign_self"
FINALIZE_TASK = "finalize_task"
CANCEL_FINALIZE = "cancel_finalize"
SEND_DM = "send_direct_message"
SHOUT = "send_shoutboard_message"
VIEW_SHOUTBOARD = "view_shoutboard"
VIEW_KB = "view_knowledge_base"
SUBMIT_TRUTH = "submit_truth"
LIST_JURY = "list_jury_submissions"
JOIN_JURY = "join_jury"
SUBMIT_VERDICT = "submit_verdict"
CANCEL_JURY = "cancel_jury"
VOTE_GOAL_REACHED = "vote_goal_reached"
REJOIN = "rejoin"
RELIEVE_SELF = "relieve_self"
MARK_DONE = "mark_integration_done"
BASH = "bash"
READ_FILE = "read_file"
WRITE_FILE = "write_file"
EDIT_FILE = "edit_file"

_COMMS = [SEND_DM, SHOUT, VIEW_SHOUTBOARD]
_WORKSPACE = [BASH, READ_FILE, WRITE_FILE, EDIT_FILE]

_BASE: dict[Situation, list[str]] = {
    Situation.IDLE: [
        WAIT,
        VIEW_BOARD,
        CREATE_TASK,
        ASSIGN_SELF,
        *_COMMS,
        VIEW_KB,
        VOTE_GOAL_REACHED,
    ],
    Situation.WAITING_FOR_SEATS: [WAIT, VIEW_BOARD, UNASSIGN_SELF, *_COMMS, VIEW_KB],
    Situation.ON_TASK: [
        WAIT,
        VIEW_BOARD,
        UNASSIGN_SELF,
        FINALIZE_TASK,
        *_COMMS,
        VIEW_KB,
        *_WORKSPACE,
    ],
    Situation.JURY: [
        WAIT,
        *_COMMS,
        VIEW_KB,
        LIST_JURY,
        SUBMIT_VERDICT,
        CANCEL_JURY,
        BASH,
        READ_FILE,
    ],
    Situation.CLOSING_TASK: [WAIT, VIEW_BOARD, *_COMMS, VIEW_KB, *_WORKSPACE, MARK_DONE],
    Situation.RELEASED: [WAIT, *_COMMS, REJOIN],
    Situation.RELIEVED: [],
    Situation.DEAD: [],
}

# joining a jury means leaving what you are doing, so it is offered only from idle
_JURY_JOINABLE = (Situation.IDLE,)
# a truth needs the KB read first, so an agent cannot duplicate or contradict one
_TRUTH_SUBMITTABLE = (
    Situation.IDLE,
    Situation.WAITING_FOR_SEATS,
    Situation.ON_TASK,
    Situation.CLOSING_TASK,
)


def tool_names(agent: Agent, harness: Harness) -> list[str]:
    names = list(_BASE[agent.situation])

    if agent.situation in _TRUTH_SUBMITTABLE and agent.seen_kb:
        names.append(SUBMIT_TRUTH)

    judgeable = [s for s in harness.kb.pending.values() if s.submitted_by != agent.name]
    if judgeable and agent.situation is not Situation.JURY:
        names.append(LIST_JURY)
        if agent.situation in _JURY_JOINABLE:
            names.append(JOIN_JURY)

    if agent.situation is Situation.ON_TASK:
        task = harness.board.get(agent.task_id or "")
        seat = task.seat_of(agent.name) if task else None
        if seat and seat.finalization:
            names.remove(FINALIZE_TASK)
            names.append(CANCEL_FINALIZE)

    if harness.force_ending and agent.situation not in (Situation.RELIEVED, Situation.DEAD):
        names.append(RELIEVE_SELF)

    return names


def tools_for(agent: Agent, harness: Harness) -> list[ToolSpec]:
    from panopticon.tools import specs  # deferred: the tool modules import the names above

    return specs(tool_names(agent, harness))

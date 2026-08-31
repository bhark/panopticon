"""An agent's own history, and the prompt built from it.

The one invariant: the rendered prompt is append-only. Turn N+1's prompt carries turn N's
prompt as a literal prefix, apart from the trailing instruction line. Every provider we
talk to caches on prefixes, and a run is hundreds of turns long, so a single reordered or
rewritten line upstream of the tail re-bills the whole transcript. Everything an entry
will ever say is frozen into `Entry.text` when it is appended; `render` only joins.

Consumed inbox items are folded in as `[inbox]` entries at the point they were consumed,
never re-emitted as a moving block at the tail, for the same reason.

Compaction is the one place the prefix is deliberately broken, and it costs a full cache
write, so it fires as late as the reserve allows and never more than three times in a row
without succeeding.
"""

from __future__ import annotations

import json

from panopticon.model import Action, Agent, Entry, QueueItem, Usage
from panopticon.providers.base import Provider

TAIL = "Take exactly one action."

# absolute buffers, not percentages: a percentage of a 1M window reserves half a run
RESERVE_TOKENS = 24_000
KEEP_TOKENS = 8_000
MAX_COMPACTION_FAILURES = 3

_TAG_WIDTH = 10
_CHARS_PER_TOKEN = 4


# append


def append_inbox(agent: Agent, items: list[QueueItem]) -> None:
    """Fold consumed queue items in at the point they were consumed."""
    for item in items:
        agent.entries.append(Entry(kind="inbox", text=item.text, turn=agent.turns))


def append_action(agent: Agent, action: Action) -> None:
    args = json.dumps(action.args, separators=(", ", ": "), sort_keys=True)
    text = f"{action.tool} {args}"
    if action.note:
        text += f"  // {action.note}"
    agent.entries.append(Entry(kind="action", text=text, turn=agent.turns))


def append_result(agent: Agent, text: str) -> None:
    agent.entries.append(Entry(kind="result", text=text, turn=agent.turns))


def reset(agent: Agent, preprompt: str) -> None:
    """Drop the transcript and start a fresh one from a deterministic preprompt."""
    agent.entries = [Entry(kind="note", text=preprompt, turn=agent.turns)]
    agent.usage.context_tokens = 0
    agent.usage.measured_entries = 0
    agent.compaction_failures = 0


# render


def render(agent: Agent) -> str:
    lines = _lines(agent.entries)
    lines.append("")
    lines.append(TAIL)
    return "\n".join(lines)


def _lines(entries: list[Entry]) -> list[str]:
    return [f"{_tag(e):<{_TAG_WIDTH}}{e.text}" for e in entries]


def _tag(entry: Entry) -> str:
    if entry.kind == "action":
        return f"[turn {entry.turn}]"
    return f"[{entry.kind}]"


# tokens


def note_usage(agent: Agent, usage: Usage) -> None:
    """Fold one turn's usage in. Call before appending that turn's action entry.

    `measured_entries` pins the reported context figure to the transcript length it was
    measured against, so the estimate below knows what is left to guess at.
    """
    agent.usage.input_tokens += usage.input_tokens
    agent.usage.output_tokens += usage.output_tokens
    agent.usage.cache_read += usage.cache_read
    agent.usage.cache_write += usage.cache_write
    agent.usage.cost_usd += usage.cost_usd
    if usage.context_tokens > 0:
        agent.usage.context_tokens = usage.context_tokens
        agent.usage.measured_entries = len(agent.entries)


def estimate_tokens(agent: Agent) -> int:
    """Provider-reported figure plus a character estimate of everything appended since."""
    reported = agent.usage.context_tokens
    if reported <= 0:
        return _estimate(agent.entries)
    return reported + _estimate(agent.entries[agent.usage.measured_entries :])


def _estimate(entries: list[Entry]) -> int:
    chars = sum(len(e.text) + _TAG_WIDTH + 1 for e in entries)
    return -(-chars // _CHARS_PER_TOKEN)


# compaction

_COMPACT_SYSTEM = """\
You are compacting your own transcript so you can keep working after it is dropped.
You have one turn. Do not call tools. Do not ask questions. Write the briefing and stop.

Your situation, for context:
{system}"""

_COMPACT_PROMPT = """\
Rewrite the transcript below as a briefing to yourself. Sections, in this order, each a \
few short lines, no prose and no praise:

1. Goal, and where it stands.
2. What you did, and what came of it.
3. What you established that is written down nowhere else.
4. Open threads: who is waiting on you, what you owe whom.
5. Your next step.

Transcript:
{transcript}"""


def needs_compaction(agent: Agent, provider: Provider) -> bool:
    return estimate_tokens(agent) > provider.context_window - RESERVE_TOKENS


async def compact(agent: Agent, provider: Provider, system: str) -> bool:
    """Replace the transcript prefix with one summary entry. False if it did not happen."""
    if agent.compaction_failures >= MAX_COMPACTION_FAILURES:
        return False
    cut = _cut_point(agent.entries)
    if cut <= 0:
        return False
    head = _lines(agent.entries[:cut])
    summary = await provider.summarize(
        _COMPACT_SYSTEM.format(system=system),
        _COMPACT_PROMPT.format(transcript="\n".join(head)),
    )
    if not summary:
        agent.compaction_failures += 1
        return False
    kept = agent.entries[cut:]
    agent.entries = [Entry(kind="summary", text=summary.strip(), turn=agent.turns), *kept]
    agent.usage.context_tokens = 0
    agent.usage.measured_entries = 0
    agent.compaction_failures = 0
    return True


def _cut_point(entries: list[Entry]) -> int:
    """First index to keep. Never lands on a result, which would orphan its action."""
    budget = KEEP_TOKENS * _CHARS_PER_TOKEN
    cut = len(entries)
    for i in range(len(entries) - 1, -1, -1):
        budget -= len(entries[i].text) + _TAG_WIDTH + 1
        if budget < 0:
            break
        cut = i
    # a window small enough that the tail alone clears the reserve would never shrink
    if cut == 0 and len(entries) > 1:
        cut = len(entries) // 2
    while cut > 0 and entries[cut].kind == "result":
        cut -= 1
    return cut

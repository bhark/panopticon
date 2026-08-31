"""An agent's own history, and the prompt built from it.

The one invariant: the rendered prompt is append-only. Turn N+1's prompt carries turn N's
prompt as a literal prefix, apart from the trailing instruction line. Every provider we
talk to caches on prefixes, and a run is hundreds of turns long, so a single reordered or
rewritten line upstream of the tail re-bills the whole transcript. Everything an entry
will ever say is frozen into `Entry.text` when it is appended; `render` only joins.

Consumed inbox items are folded in as `[inbox]` entries at the point they were consumed,
never re-emitted as a moving block at the tail, for the same reason.

Compaction is one of two places the prefix is deliberately broken, and it costs a full
cache write, so it fires as late as the reserve allows. The other is `trim_cold`, which
only runs once the provider's cache has certainly expired and the rewrite is happening
anyway.
"""

from __future__ import annotations

import json
import time

from panopticon.model import Action, Agent, Entry, QueueItem, Usage
from panopticon.providers.base import Provider

TAIL = "Take exactly one action."

# absolute buffers, not percentages: a percentage of a 1M window reserves half a run
RESERVE_TOKENS = 24_000
KEEP_TOKENS = 8_000
MAX_COMPACTION_FAILURES = 3

# models degrade well before a very large window runs out; nothing grows past this
SOFT_WINDOW = 400_000

# the server-side prompt cache tops out at an hour, so past it the rewrite is certain
COLD_CACHE_SECONDS = 3600
KEEP_RECENT_RESULTS = 5
CLEARED = "[old result cleared]"
# results an agent can simply fetch again; a dm or a jury notice exists nowhere else
RE_READABLE = frozenset(
    {"bash", "read_file", "view_task_board", "view_shoutboard", "view_knowledge_base"}
)

UNCHANGED = "Nothing new since you last looked."

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


def append_result(agent: Agent, text: str, tool: str = "") -> None:
    agent.entries.append(Entry(kind="result", text=text, turn=agent.turns, tool=tool))


def dedupe_result(agent: Agent, tool: str, text: str) -> str:
    """Collapse a re-read that would append an exact copy of what the transcript still holds.

    Scans the transcript rather than caching per agent, so a cleared, compacted or reset
    entry answers in full again on its own.
    """
    for entry in reversed(agent.entries):
        if entry.kind == "result" and entry.tool == tool and entry.text != UNCHANGED:
            return UNCHANGED if entry.text == text else text
    return text


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


def estimate_tokens(agent: Agent, system: str = "") -> int:
    """Provider-reported figure plus a character estimate of everything appended since.

    The reported figure already covers the system prompt; the fallback estimate does not,
    and kimi reports nothing at all, so for that provider it is the only figure there is.
    """
    reported = agent.usage.context_tokens
    if reported <= 0:
        return _estimate(agent.entries) + -(-len(system) // _CHARS_PER_TOKEN)
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


def usable_window(provider: Provider) -> int:
    return min(provider.context_window, SOFT_WINDOW)


def needs_compaction(agent: Agent, provider: Provider, system: str = "") -> bool:
    return estimate_tokens(agent, system) > usable_window(provider) - RESERVE_TOKENS


async def compact(
    agent: Agent,
    provider: Provider,
    system: str,
    *,
    fallbacks: tuple[Provider, ...] = (),
    log_path: str = "",
) -> str:
    """Replace the transcript prefix with one entry. Returns how, or "" if there was nothing to do.

    Dropping the prefix outright is the floor. A summary is better, but a provider that is
    down cannot write one, and leaving the transcript over the window would only feed the
    provider a prompt it must reject until the agent dies of it.
    """
    cut = _cut_point(agent.entries)
    if cut <= 0:
        return ""
    summary = ""
    if agent.compaction_failures < MAX_COMPACTION_FAILURES:
        summary = await _summarize(agent, (provider, *fallbacks), system, cut)
    replacement = (
        Entry(kind="summary", text=summary, turn=agent.turns)
        if summary
        else Entry(kind="note", text=_dropped(cut, log_path), turn=agent.turns)
    )
    agent.entries = [replacement, *agent.entries[cut:]]
    agent.usage.context_tokens = 0
    agent.usage.measured_entries = 0
    if summary:
        agent.compaction_failures = 0
    return "summarized" if summary else "dropped"


async def _summarize(agent: Agent, providers: tuple[Provider, ...], system: str, cut: int) -> str:
    head = "\n".join(_lines(agent.entries[:cut]))
    for provider in providers:
        summary = await provider.summarize(
            _COMPACT_SYSTEM.format(system=system),
            _COMPACT_PROMPT.format(transcript=head),
        )
        if summary and summary.strip():
            return summary.strip()
    agent.compaction_failures += 1
    return ""


def _dropped(cut: int, log_path: str) -> str:
    text = (
        f"The first {cut} entries of your transcript were dropped to make room and could not "
        "be summarized. Work from what is left."
    )
    return f"{text} Your full history is at {log_path}." if log_path else text


def trim_cold(agent: Agent) -> int:
    """Blank re-readable result bodies once the provider's cache has certainly expired.

    Breaking the prefix costs a cache write, so this only fires past the cache TTL, where
    that write is already going to happen. Earlier it would manufacture the cost it saves.
    """
    if not agent.last_turn_at or time.time() - agent.last_turn_at < COLD_CACHE_SECONDS:
        return 0
    seen, cleared = 0, 0
    for entry in reversed(agent.entries):
        if entry.kind != "result":
            continue
        seen += 1
        if seen <= KEEP_RECENT_RESULTS or entry.tool not in RE_READABLE:
            continue
        if entry.text != CLEARED:
            entry.text = CLEARED
            cleared += 1
    if cleared:
        agent.usage.context_tokens = 0
        agent.usage.measured_entries = 0
    return cleared


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

"""Every string an agent is shown. Kept in one place so tone stays consistent."""

from __future__ import annotations

from panopticon.model import Agent, Harness, ToolSpec


def build_system_prompt(agent: Agent, harness: Harness, tools: list[ToolSpec]) -> str:
    """Identity, goal, brevity rules, the output contract, situation, available tools.

    Stable for as long as the agent's situation and tool set are, so it caches.
    """
    ...


def situation_preprompt(agent: Agent, harness: Harness, note: str) -> str:
    """The deterministic opening line of a freshly cleared transcript."""
    ...

"""The tools an agent can call. Importing the modules is what fills the registry."""

from __future__ import annotations

from panopticon.tools import (  # noqa: F401
    board,
    comms,
    jury,
    knowledge,
    lifecycle,
    wait,
    workspace,
)
from panopticon.tools.registry import REGISTRY, dispatch, tools_for

__all__ = ["REGISTRY", "dispatch", "tools_for"]

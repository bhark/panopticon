"""Noticing that a newer panopticon exists, and installing it.

The check never sits on the critical path: the interface paints whatever the cache already
says, and the refresh runs as a task beside the harness. Releases are public, so this needs
no credential - plain HTTPS against the GitHub API.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

from panopticon import __version__
from panopticon import config as config_mod

REPO = "bhark/panopticon"
API = f"https://api.github.com/repos/{REPO}/releases"
INTERVAL = 24 * 60 * 60
TIMEOUT = 10

CACHE_FILE = config_mod.CONFIG_DIR / "update.json"


class UpdateError(Exception):
    pass


# the cache


def remember(tag: str) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"checked_at": time.time(), "tag": tag}))


def _recall() -> tuple[str, float]:
    """The tag last seen and when we looked - a cache we can't read is just a cache miss."""
    try:
        data = json.loads(CACHE_FILE.read_text())
        return str(data["tag"]), float(data["checked_at"])
    except OSError, ValueError, KeyError, TypeError:
        return "", 0.0


# comparing


def _number(version: str) -> tuple[int, ...] | None:
    parts = version.removeprefix("v").split(".")
    return tuple(int(p) for p in parts) if all(p.isdigit() for p in parts) else None


def behind(tag: str) -> bool:
    """Anything that isn't plain numbers on both sides is not worth nudging anyone about."""
    here, there = _number(__version__), _number(tag)
    return bool(here and there and there > here)


def note() -> str:
    """The one line shown when a newer release is out, or empty."""
    tag, _ = _recall()
    return f"update {tag.removeprefix('v')} available · panopticon update" if behind(tag) else ""


# the network


def _release(tag: str = "") -> dict:
    url = f"{API}/tags/{tag}" if tag else f"{API}/latest"
    request = urllib.request.Request(url, headers={"User-Agent": f"panopticon/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # the feed 404s both for a repo with no releases at all and for an unknown tag
        if exc.code == 404:
            raise UpdateError(
                f"{REPO} has no release {tag}" if tag else f"{REPO} has no releases yet"
            ) from exc
        raise UpdateError(f"the release feed answered {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise UpdateError(f"could not reach the release feed: {exc}") from exc


def latest() -> str:
    tag = _release().get("tag_name")
    if not tag:
        raise UpdateError(f"{REPO} has no releases yet")
    return str(tag)


async def refresh() -> None:
    """Learn today's answer for tomorrow. Never awaited by anything the human is waiting on."""
    if not sys.stdout.isatty():
        return
    tag, checked_at = _recall()
    if time.time() - checked_at < INTERVAL:
        return
    # a check that fails keeps yesterday's answer but still counts as today's attempt
    with contextlib.suppress(UpdateError):
        tag = await asyncio.to_thread(latest)
    remember(tag)


def install(tag: str) -> None:
    assets = [a for a in _release(tag).get("assets", []) if a["name"].endswith(".whl")]
    if not assets:
        raise UpdateError(f"release {tag} has no wheel attached - install by hand from {REPO}")
    url = assets[0]["browser_download_url"]
    # uv rewrites the tool's own venv; this process keeps running off bytecode it already has
    result = subprocess.run(["uv", "tool", "install", "--force", f"panopticon @ {url}"])
    if result.returncode:
        raise UpdateError("uv could not install the new version")

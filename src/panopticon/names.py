"""Short, pronounceable names. Agents know each other by nothing else."""

from __future__ import annotations

import random

_ONSET = ["b", "br", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"]
_MID = ["b", "d", "l", "m", "n", "r", "s", "t", "v", "z", "ll", "nn", "rr", "sh", "th"]
_VOWEL = ["a", "e", "i", "o", "u"]
_END = ["", "", "", "n", "s", "r", "l"]


def _name(rng: random.Random) -> str:
    return (
        rng.choice(_ONSET)
        + rng.choice(_VOWEL)
        + rng.choice(_MID)
        + rng.choice(_VOWEL)
        + rng.choice(_END)
    )


def generate(count: int, rng: random.Random | None = None) -> list[str]:
    rng = rng or random.Random()
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < count:
        word = _name(rng)
        if not 4 <= len(word) <= 7 or word in seen:
            continue
        seen.add(word)
        names.append(word.capitalize())
    return names

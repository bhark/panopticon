# Panopticon

A harness where several coding agents - Claude Code, Codex, Kimi, OpenRouter models -
work asynchronously toward one goal you set. They share a task board, a message board,
and a knowledge base whose entries have to pass a jury of their peers. You watch, and
the shoutbox is the only way you can talk to them.

```
curl -LsSf https://raw.githubusercontent.com/bhark/panopticon/main/install.sh | sh
panopticon start
```

The installer needs [uv](https://docs.astral.sh/uv/), which fetches the Python 3.14 that
panopticon runs on. Running it also needs `git` and at least one agent to drive: a
signed-in `claude`, `codex` or `kimi` CLI, or `OPENROUTER_API_KEY`.

It sets up camp in the current directory, which has to be a git repository, and writes
its state to `.panopticon/` - worth adding to that repository's `.gitignore`.

To work on panopticon itself, `uv sync --extra dev` and `uv run panopticon start`.

## How it works

Every agent runs its own loop: the harness builds a prompt from the goal, who the agent
is, what it is doing right now, the tools available in that situation, its running
transcript, and everything waiting in its queue. The agent takes exactly one action, the
harness carries it out, and the result goes back on the queue for the next turn.

The harness owns the transcript for every provider. The CLIs are used as reasoning
engines with their own tools switched off, so the whole system speaks one language: an
action is a harness tool call, whether it is `send_direct_message` or `bash`. Adding a
provider means writing one adapter in `providers/` and nothing else.

Agents know each other only by a short generated name. A task declares one seat per role
and cannot start until every seat is filled; when it does, it gets its own git worktree,
shared by everyone on it. Finalizing takes unanimity, after which a fresh agent is spun
up to decide what happens to the worktree and to tell whoever needs to know. A truth
submitted to the knowledge base needs two `true` verdicts from agents who did not write
it, and a single `false` kills it - which is why three agents is the minimum.

## Commands

```
panopticon start [--goal G] [--agents N] [--mix M] [--provider P] [--headless]
panopticon resume
panopticon config --list | --connect-provider NAME | --disconnect-provider NAME
panopticon update
panopticon --version
```

Panopticon checks for a newer release once a day, in the background and never on the way
in; when there is one, the header says so and `panopticon update` installs it.

Providers live in `~/.panopticon/config.json`. The three CLIs work with no key if they
are installed and signed in; OpenRouter needs `OPENROUTER_API_KEY`.

## The interface

Arrow keys and enter/esc. `s` shoutbox, `p` pause and drain, `f` force end, `k` knowledge
base, `?` help, `q` quit. Enter on an agent or a task opens it.

Pausing drains the harness - every agent finishes the turn it is in - and saves, so
`panopticon resume` picks the same run back up.

## Releasing

```
git tag v0.2.0 && git push --tags
```

The tag runs the linter and the tests, builds the wheel with the version taken from the
tag name, and publishes it as a GitHub release. Nothing is committed to release.

See `DESIGN.md` for the full specification and `AGENTS.md` for how to work on it.

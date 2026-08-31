from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from panopticon.model import Action, ArgSpec, ToolSpec
from panopticon.providers import _cli, api_openrouter, cli_claude, cli_codex, cli_kimi, registry
from panopticon.providers.base import TurnRequest, parse_action, strict_action_schema
from panopticon.providers.mock import MockProvider

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


async def nothing(ctx, args):  # a ToolSpec needs a handler; none of these tests call one
    raise AssertionError


TOOLS = [
    ToolSpec(
        "wait", "Do nothing.", {"minutes": ArgSpec("int", "How long.", required=False)}, nothing
    ),
    ToolSpec(
        "send_direct_message", "Message one agent.", {"to": ArgSpec("string", "Who.")}, nothing
    ),
]


def request(prompt: str = "[turn 1] nothing waiting") -> TurnRequest:
    return TurnRequest(agent="Suzanne", system="You are Suzanne.", prompt=prompt, tools=TOOLS)


class TestParseAction:
    def test_bare_object(self):
        action, error = parse_action('{"tool": "wait", "args": {"minutes": 5}}', TOOLS)
        assert error is None
        assert action == Action(tool="wait", args={"minutes": 5})

    def test_fenced_json(self):
        text = 'Here you go:\n```json\n{"tool": "wait", "args": {}}\n```\n'
        action, error = parse_action(text, TOOLS)
        assert error is None and action.tool == "wait"

    def test_prose_then_json_with_nested_braces(self):
        text = (
            "I will message Alex.\n"
            '{"tool": "send_direct_message", "args": {"to": "Alex", "meta": {"a": {"b": 1}}}}'
        )
        action, error = parse_action(text, TOOLS)
        assert error is None
        assert action.args["meta"] == {"a": {"b": 1}}

    def test_prose_containing_a_decoy_object_takes_the_last_one(self):
        text = (
            'Not this one: {"tool": "wait"} '
            'but this: {"tool": "send_direct_message", "args": {"to": "Alex"}}'
        )
        action, error = parse_action(text, TOOLS)
        assert error is None and action.tool == "send_direct_message"

    def test_args_as_a_json_string_from_strict_mode(self):
        action, error = parse_action(
            '{"tool": "wait", "args": "{\\"minutes\\": 5}", "note": ""}', TOOLS
        )
        assert error is None
        assert action.args == {"minutes": 5}

    def test_args_as_an_unparseable_string(self):
        action, error = parse_action('{"tool": "wait", "args": "minutes=5"}', TOOLS)
        assert action is None and "not valid JSON" in error

    def test_args_as_an_empty_string(self):
        action, error = parse_action('{"tool": "wait", "args": ""}', TOOLS)
        assert error is None and action.args == {}

    def test_a_json_object_that_is_not_an_action(self):
        action, error = parse_action('{"thinking": "hmm", "next": "wait"}', TOOLS)
        assert action is None and "no 'tool' field" in error

    def test_an_unknown_tool_name_lists_what_is_available(self):
        action, error = parse_action('{"tool": "rm_rf", "args": {}}', TOOLS)
        assert action is None
        assert "rm_rf" in error and "send_direct_message" in error

    def test_args_that_are_not_an_object(self):
        action, error = parse_action('{"tool": "wait", "args": [1, 2]}', TOOLS)
        assert action is None and "not an object" in error

    def test_prose_with_no_json_at_all(self):
        action, error = parse_action("I think I should wait a while.", TOOLS)
        assert action is None and "no JSON object" in error

    def test_empty_response(self):
        action, error = parse_action("   \n ", TOOLS)
        assert action is None and error == "empty response"


class TestStrictSchema:
    def test_every_object_is_closed_and_every_property_required(self):
        """codex rejects the permissive schema outright: invalid_json_schema on `args`."""
        schema = strict_action_schema(TOOLS)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert schema["properties"]["args"]["type"] == "string"
        assert schema["properties"]["tool"]["enum"] == ["wait", "send_direct_message"]


class TestClaudeCLI:
    def test_parses_a_real_success_stream(self):
        response = cli_claude.parse_output(fixture("claude_success.ndjson"), TOOLS)
        assert response.error is None
        assert response.action == Action(
            tool="wait", args={"minutes": 1}, note="Waiting for tasks or test failure information"
        )
        assert response.usage.input_tokens == 1079
        assert response.usage.output_tokens == 390
        assert response.usage.context_tokens == 1469
        assert response.usage.cost_usd == pytest.approx(0.003992)

    def test_an_api_error_is_an_error_even_though_subtype_says_success(self):
        """A failed turn still reports subtype=success; only is_error tells the truth."""
        stdout = fixture("claude_api_error.ndjson")
        assert json.loads(stdout.splitlines()[-1])["subtype"] == "success"
        response = cli_claude.parse_output(stdout, TOOLS)
        assert response.action is None
        assert "api_error" in response.error
        assert "nonexistent-model-xyz" in response.error

    def test_a_stream_with_no_result_event_is_not_a_parse_failure(self):
        assert cli_claude.parse_output('{"type":"system","subtype":"init"}', TOOLS) is None

    def test_junk_lines_between_events_are_skipped(self):
        stdout = "warning: something\n" + fixture("claude_success.ndjson") + "\nnot json\n"
        assert cli_claude.parse_output(stdout, TOOLS).action is not None

    def test_the_invocation_keeps_every_load_bearing_flag(self):
        argv = cli_claude.ClaudeCLI("claude", model="haiku")._argv("sys", "hi", {"type": "object"})
        assert argv[argv.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in argv  # stream-json hard errors without it
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--setting-sources") + 1] == ""  # or the user's hooks leak in
        assert "--strict-mcp-config" in argv  # --tools "" does not drop MCP tools
        assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
        assert "--no-session-persistence" in argv

    async def test_a_missing_binary_comes_back_as_an_error_not_an_exception(self):
        provider = cli_claude.ClaudeCLI("claude", model="haiku", bin="claude-does-not-exist")
        response = await provider.act(request())
        assert response.action is None
        assert "could not start" in response.error


class TestProcess:
    async def test_a_timeout_kills_the_children_the_cli_spawned_too(self, tmp_path):
        """The CLIs fork; killing only the wrapper leaves a model call running and billing."""
        marker = tmp_path / "survived"
        script = f"(sleep 0.4; touch {marker}) & wait"
        done = await _cli.run(["sh", "-c", script], timeout=0.1)

        assert "timed out" in done.error
        await asyncio.sleep(0.8)
        assert not marker.exists()

    async def test_a_binary_that_is_not_there_is_reported_not_raised(self):
        done = await _cli.run(["panopticon-no-such-binary"])
        assert done.error and "could not start" in done.error

    def test_ndjson_skips_anything_that_is_not_a_json_object(self):
        stream = 'warning: x\n{"a": 1}\n[1,2]\n{bad}\n\n{"b": 2}'
        assert list(_cli.ndjson(stream)) == [{"a": 1}, {"b": 2}]


class TestCodexCLI:
    def test_parses_a_real_success_stream_including_the_string_encoded_args(self):
        response = cli_codex.parse_output(fixture("codex_success.ndjson"), TOOLS)
        assert response.error is None
        assert response.action.tool == "wait"
        assert response.action.args == {"minutes": 1}
        assert response.usage.input_tokens == 17397
        assert response.usage.context_tokens == 17434

    def test_a_turn_failure_is_an_error(self):
        response = cli_codex.parse_output(fixture("codex_turn_failed.ndjson"), TOOLS, code=1)
        assert response.action is None
        assert "not supported when using Codex" in response.error

    def test_an_item_level_warning_alongside_a_good_message_is_not_an_error(self):
        stdout = (
            '{"type":"item.completed","item":'
            '{"type":"error","message":"Model metadata not found"}}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":'
            '"{\\"tool\\":\\"wait\\",\\"args\\":\\"{}\\",\\"note\\":\\"\\"}"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}'
        )
        response = cli_codex.parse_output(stdout, TOOLS)
        assert response.error is None and response.action.tool == "wait"

    def test_an_item_level_warning_with_nothing_else_does_surface(self):
        stdout = (
            '{"type":"item.completed","item":{"type":"error","message":"Model metadata not found"}}'
        )
        assert "Model metadata" in cli_codex.parse_output(stdout, TOOLS).error

    def test_the_invocation_passes_the_prompt_as_argv_and_prepends_the_system_prompt(self):
        """There is no --append-system-prompt, and a piped stdin arrives as a <stdin> block."""
        argv = cli_codex.CodexCLI("codex", model="gpt-5.1")._argv(
            "SYS\n\nUSER", "/tmp", "/tmp/s.json"
        )
        assert argv[-1] == "SYS\n\nUSER"
        assert argv[argv.index("-C") + 1] == "/tmp"
        assert argv[argv.index("--output-schema") + 1] == "/tmp/s.json"
        assert {"--ephemeral", "--ignore-user-config", "--skip-git-repo-check"} <= set(argv)

    async def test_a_missing_binary_comes_back_as_an_error(self):
        provider = cli_codex.CodexCLI("codex", model="x", bin="codex-does-not-exist")
        response = await provider.act(request())
        assert response.action is None and "could not start" in response.error


class TestKimiCLI:
    def test_parses_a_real_success_stream(self):
        response = cli_kimi.parse_output(fixture("kimi_success.ndjson"), TOOLS)
        assert response.error is None
        assert response.action.tool == "wait" and response.action.args == {"minutes": 5}

    def test_a_wrong_shape_is_a_parse_error_not_a_crash(self):
        """Left without the shape reminder, kimi invents its own keys. Captured, not imagined."""
        response = cli_kimi.parse_output(fixture("kimi_wrong_shape.ndjson"), TOOLS)
        assert response.action is None
        assert "no 'tool' field" in response.error
        assert response.raw == '{"action": "wait", "minutes": 5}'

    def test_meta_lines_are_ignored(self):
        assert cli_kimi.assistant_text(fixture("kimi_success.ndjson")).startswith('{"tool"')

    async def test_a_parse_failure_reprompts_exactly_once_then_gives_up(self):
        prompts: list[str] = []

        async def fake_run(argv, *, cwd=None, timeout=0.0):
            prompts.append(argv[argv.index("-p") + 1])
            return _cli.Completed(fixture("kimi_wrong_shape.ndjson"), "", 0)

        provider = cli_kimi.KimiCLI("kimi", model="k3")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_cli, "run", fake_run)
            response = await provider.act(request())

        assert len(prompts) == 2
        assert cli_kimi.SHAPE in prompts[0]
        assert "could not be used" in prompts[1]
        assert prompts[1].startswith(prompts[0])
        assert response.action is None and response.error is not None

    async def test_a_good_reply_never_reprompts(self):
        calls = 0

        async def fake_run(argv, *, cwd=None, timeout=0.0):
            nonlocal calls
            calls += 1
            return _cli.Completed(fixture("kimi_success.ndjson"), "", 0)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_cli, "run", fake_run)
            response = await cli_kimi.KimiCLI("kimi", model="k3").act(request())

        assert calls == 1 and response.action.tool == "wait"


class TestOpenRouter:
    def _provider(self, handler, **kwargs) -> api_openrouter.OpenRouter:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return api_openrouter.OpenRouter("or", model="m", backoff=0.0, client=client, **kwargs)

    def _ok(self, content: str) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 40,
                    "total_tokens": 1240,
                    "prompt_tokens_details": {"cached_tokens": 1000},
                    "cost": 0.0012,
                },
            },
        )

    async def test_a_missing_key_is_an_error_not_a_request(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        def handler(request):
            raise AssertionError("should not have been called")

        response = await self._provider(handler).act(request())
        assert response.action is None and "OPENROUTER_API_KEY is not set" in response.error

    async def test_sends_one_system_and_one_user_message_with_the_strict_schema(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        seen = {}

        def handler(req):
            seen.update(json.loads(req.content))
            assert req.headers["authorization"] == "Bearer k"
            return self._ok('{"tool": "wait", "args": "{}", "note": ""}')

        response = await self._provider(handler).act(request())
        assert response.action == Action(tool="wait", args={}, note="")
        assert [m["role"] for m in seen["messages"]] == ["system", "user"]
        assert seen["response_format"]["json_schema"]["strict"] is True
        assert (
            seen["response_format"]["json_schema"]["schema"]["properties"]["args"]["type"]
            == "string"
        )
        assert response.usage.context_tokens == 1240
        assert response.usage.cache_read == 1000
        assert response.usage.cost_usd == pytest.approx(0.0012)

    async def test_retries_a_429_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        codes = [429, 503, 200]

        def handler(req):
            code = codes.pop(0)
            if code == 200:
                return self._ok('{"tool": "wait", "args": "{}", "note": ""}')
            return httpx.Response(code, text="slow down")

        response = await self._provider(handler).act(request())
        assert response.action is not None and codes == []

    async def test_gives_up_after_the_attempt_budget(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        attempts = 0

        def handler(req):
            nonlocal attempts
            attempts += 1
            return httpx.Response(429, text="slow down")

        response = await self._provider(handler, attempts=3).act(request())
        assert attempts == 3
        assert response.action is None and "429" in response.error

    async def test_a_400_is_not_retried(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        attempts = 0

        def handler(req):
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, text="bad model")

        response = await self._provider(handler).act(request())
        assert attempts == 1 and "400" in response.error

    async def test_a_transport_error_is_an_error_not_an_exception(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")

        def handler(req):
            raise httpx.ConnectError("no route to host")

        response = await self._provider(handler, attempts=2).act(request())
        assert response.action is None and "request failed" in response.error


class TestMock:
    async def test_replays_a_script_per_agent(self):
        provider = MockProvider(
            {
                "Suzanne": [Action("wait", {}), Action("send_direct_message", {"to": "Alex"})],
                "Alex": [Action("wait", {"minutes": 5})],
            }
        )
        assert (await provider.act(TurnRequest("Suzanne", "s", "p", TOOLS))).action.tool == "wait"
        assert (await provider.act(TurnRequest("Alex", "s", "p", TOOLS))).action.args == {
            "minutes": 5
        }
        second = await provider.act(TurnRequest("Suzanne", "s", "p", TOOLS))
        assert second.action.tool == "send_direct_message"

    async def test_an_exhausted_script_is_an_error_naming_the_agent(self):
        provider = MockProvider({"Suzanne": []})
        response = await provider.act(TurnRequest("Suzanne", "s", "p", TOOLS))
        assert response.action is None and "Suzanne" in response.error

    async def test_a_callable_script_sees_the_prompt(self):
        provider = MockProvider(lambda req: Action("wait", {"saw": req.prompt[:5]}))
        response = await provider.act(TurnRequest("Suzanne", "s", "hello world", TOOLS))
        assert response.action.args == {"saw": "hello"}
        assert provider.calls[0].prompt == "hello world"


class TestRegistry:
    def test_builds_each_kind_with_its_own_default_window(self):
        assert (
            registry.build("claude", {"kind": "claude_cli", "model": "opus"}).context_window
            == 200_000
        )
        assert registry.build("kimi", {"kind": "kimi_cli", "model": "k3"}).context_window == 262_144
        assert registry.build("mock", {"kind": "mock"}).context_window == 200_000

    def test_config_overrides_the_default_window_and_the_binary(self):
        provider = registry.build(
            "claude",
            {
                "kind": "claude_cli",
                "model": "opus",
                "context_window": 900_000,
                "bin": "/opt/claude",
            },
        )
        assert provider.context_window == 900_000
        assert provider.bin == "/opt/claude"
        assert provider.key == "claude"

    def test_api_only_settings_do_not_leak_into_a_cli_adapter(self):
        provider = registry.build(
            "claude",
            {"kind": "claude_cli", "model": "opus", "api_key_env": "NOPE", "base_url": "http://x"},
        )
        assert provider.model == "opus"

    def test_an_unknown_kind_names_the_ones_that_exist(self):
        with pytest.raises(ValueError, match="openai_cli"):
            registry.build("x", {"kind": "openai_cli", "model": "m"})

    def test_a_missing_model_is_caught_at_config_time(self):
        with pytest.raises(ValueError, match="no model"):
            registry.build("x", {"kind": "openrouter"})

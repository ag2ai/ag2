# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for Kosa8Environment against a fake kosa8 Cloud API."""

import json
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from ag2 import Context, Variable
from ag2.extensions.kosa8 import Kosa8Environment
from ag2.extensions.kosa8.environment import make_client
from ag2.extensions.kosa8.sandbox import Kosa8Sandbox
from ag2.tools import SandboxCodeTool, SandboxShellTool
from ag2.tools.sandbox import ExecResult, SandboxFactory, WorkdirAware

from .fake_cloud import FakeCloud, error_answer, exec_answer

KEY = "k8c_live_test"  # pragma: allowlist secret


class _Clients:
    """Stands in for ``make_client``: one fake cloud, and the arguments of every call."""

    def __init__(self, cloud: FakeCloud) -> None:
        self.cloud = cloud
        self.args: list[tuple[str | None, str | None]] = []

    def __call__(self, api_key: str | None, base_url: str | None) -> httpx.AsyncClient:
        self.args.append((api_key, base_url))
        return self.cloud.client()


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch) -> FakeCloud:
    fake = FakeCloud()
    monkeypatch.setattr("ag2.extensions.kosa8.environment.make_client", _Clients(fake))
    return fake


def _clients(monkeypatch: pytest.MonkeyPatch, cloud: FakeCloud) -> _Clients:
    clients = _Clients(cloud)
    monkeypatch.setattr("ag2.extensions.kosa8.environment.make_client", clients)
    return clients


def _create_body(cloud: FakeCloud, index: int = 0) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(cloud.calls("POST", "/v1/sandboxes")[index].content)
    return body


def test_satisfies_factory_protocols() -> None:
    env = Kosa8Environment(api_key=KEY)
    assert isinstance(env, SandboxFactory)
    assert isinstance(env, WorkdirAware)
    assert env.workdir == PurePosixPath("/workspace")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeout": 0}, "timeout"),
        ({"start_timeout": 0}, "start_timeout"),
        ({"ttl_s": 59}, "ttl_s"),
        ({"ttl_s": 7_776_001}, "ttl_s"),
        ({"workdir": "rel"}, "absolute"),
    ],
)
def test_invalid_settings_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Kosa8Environment(**kwargs)


class TestMakeClient:
    def test_reads_key_and_url_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KOSA8_API_KEY", KEY)
        monkeypatch.setenv("KOSA8_API_URL", "https://kosa8.example.com/")
        client = make_client(None, None)
        assert client.base_url == "https://kosa8.example.com"
        assert client.headers["Authorization"] == f"Bearer {KEY}"

    def test_defaults_to_kosa8_cloud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KOSA8_API_URL", raising=False)
        assert make_client(KEY, None).base_url == "https://api.kosa8.com"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KOSA8_API_KEY", raising=False)
        with pytest.raises(ValueError, match="KOSA8_API_KEY"):
            make_client(None, None)

    def test_plain_http_only_for_localhost(self) -> None:
        with pytest.raises(ValueError, match="https"):
            make_client(KEY, "http://kosa8.example.com")
        assert make_client(KEY, "http://localhost:8787").base_url == "http://localhost:8787"


@pytest.mark.asyncio
class TestOpen:
    async def test_open_yields_a_sandbox_with_a_bounded_lifetime(self, cloud: FakeCloud) -> None:
        async with Kosa8Environment(api_key=KEY) as env, env.open() as sandbox:
            assert isinstance(sandbox, Kosa8Sandbox)
            assert sandbox.sandbox_id == "sbx_1"

        assert _create_body(cloud) == {"template": "python", "ttl_s": 3600}
        assert json.loads(cloud.calls("PUT", "/labels")[0].content) == {"labels": {"created-by": "ag2"}}
        assert cloud.calls("PUT", "/egress") == []
        assert cloud.deleted == ["sbx_1"]

    async def test_every_option_reaches_the_create(self, cloud: FakeCloud) -> None:
        env = Kosa8Environment(
            api_key=KEY,
            template="node",
            region="us1",
            name="agent-box",
            labels={"team": "data", "created-by": "me"},
            egress={"internet": False, "allow": ["pypi.org"]},
            persistent=True,
            ttl_s=None,
        )
        async with env, env.open():
            pass

        assert _create_body(cloud) == {"template": "node", "region": "us1", "name": "agent-box", "persistent": True}
        assert json.loads(cloud.calls("PUT", "/labels")[0].content) == {"labels": {"team": "data", "created-by": "ag2"}}
        assert json.loads(cloud.calls("PUT", "/egress")[0].content) == {"internet": False, "allow": ["pypi.org"]}

    async def test_env_vars_reach_every_command_under_per_call_env(self, cloud: FakeCloud) -> None:
        async with Kosa8Environment(api_key=KEY, env_vars={"A": "1", "B": "2"}) as env, env.open() as sandbox:
            await sandbox.exec(["env"], env={"B": "call"})
        assert cloud.exec_bodies()[-1]["env"] == ["A=1", "B=call"]

    async def test_open_resolves_variables_from_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cloud = FakeCloud()
        clients = _clients(monkeypatch, cloud)
        env = Kosa8Environment(
            api_key=Variable("tenant_key"),
            base_url=Variable(),
            template=Variable("tpl"),
            name=Variable(),
            labels=Variable(),
        )
        context = Context(
            stream=MagicMock(),
            variables={
                "tenant_key": KEY,
                "base_url": "https://kosa8.example.com",
                "tpl": "base",
                "name": "tenant-a",
                "labels": {"tenant": "a"},
            },
        )
        async with env, env.open(context):
            pass
        assert clients.args == [(KEY, "https://kosa8.example.com")]
        assert _create_body(cloud) == {"template": "base", "name": "tenant-a", "ttl_s": 3600}

    async def test_open_missing_variable_raises_key_error(self, cloud: FakeCloud) -> None:
        env = Kosa8Environment(api_key=Variable("absent"))
        context = Context(stream=MagicMock(), variables={})
        with pytest.raises(KeyError):
            async with env.open(context):
                pass

    async def test_missing_context_for_variable_raises(self, cloud: FakeCloud) -> None:
        with pytest.raises(RuntimeError, match="no Context"):
            async with Kosa8Environment(api_key=Variable("k")).open():
                pass
        assert cloud.created == 0

    async def test_invalid_env_var_name_is_refused_before_a_client_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clients = _clients(monkeypatch, FakeCloud())
        with pytest.raises(ValueError, match="environment variable name"):
            async with Kosa8Environment(api_key=KEY, env_vars={"BAD-NAME": "x"}).open():
                pass
        assert clients.args == []

    async def test_create_error_propagates(self, cloud: FakeCloud) -> None:
        cloud.create_error = error_answer(401, "unauthorized", "invalid API key")
        env = Kosa8Environment(api_key=KEY)
        with pytest.raises(Exception, match="invalid API key"):
            async with env.open():
                pass
        await env.aclose()

    async def test_open_reuses_sandbox_until_environment_close(self, cloud: FakeCloud) -> None:
        env = Kosa8Environment(api_key=KEY)
        async with env.open() as first:
            pass
        async with env.open() as second:
            pass
        assert first is second
        assert cloud.created == 1
        assert cloud.deleted == []
        await env.aclose()
        assert cloud.deleted == ["sbx_1"]

        async with env.open() as third:
            assert third is not first
        await env.aclose()
        assert cloud.created == 2

    async def test_each_tenant_gets_its_own_sandbox(self, cloud: FakeCloud) -> None:
        env = Kosa8Environment(api_key=KEY, name=Variable())
        for tenant in ("a", "b", "a"):
            context = Context(stream=MagicMock(), variables={"name": tenant})
            async with env.open(context):
                pass
        assert cloud.created == 2
        await env.aclose()
        assert sorted(cloud.deleted) == ["sbx_1", "sbx_2"]

    async def test_close_retries_a_sandbox_it_failed_to_delete(self, cloud: FakeCloud) -> None:
        cloud.delete_errors = [error_answer(502, "host_failed")]
        env = Kosa8Environment(api_key=KEY)
        async with env.open():
            pass
        await env.aclose()
        assert cloud.deleted == []
        await env.aclose()
        assert cloud.deleted == ["sbx_1"]

    async def test_shell_tool_runs_in_the_sandbox(self, cloud: FakeCloud) -> None:
        cloud.exec_results = [exec_answer("hello\n")]
        env = Kosa8Environment(api_key=KEY)
        tool = SandboxShellTool(env)
        async with env, env.open() as sandbox:
            result = await sandbox.exec(["sh", "-c", "echo hello"])
        assert result == ExecResult(output="hello", exit_code=0)
        assert isinstance(deepcopy(tool), SandboxShellTool)

    async def test_code_tool_runs_python(self, cloud: FakeCloud) -> None:
        cloud.exec_results = [exec_answer("ok\n")]
        env = Kosa8Environment(api_key=KEY)
        result = await SandboxCodeTool(env).environment.run("print('ok')", "python")
        await env.aclose()
        assert result.output == "ok"
        assert cloud.exec_bodies()[-1]["cmd"][5:] == ["python", "-c", "print('ok')"]

    async def test_code_environment_runs_bash_with_sh(self, cloud: FakeCloud) -> None:
        cloud.exec_results = [exec_answer("hi\n")]
        env = Kosa8Environment(api_key=KEY)
        result = await env.code_environment().run("echo hi", "bash")
        await env.aclose()
        assert result.output == "hi"
        assert cloud.exec_bodies()[-1]["cmd"][5:] == ["sh", "-c", "echo hi"]


class TestDeepcopy:
    def test_environment_deepcopy_returns_same_instance(self) -> None:
        env = Kosa8Environment(api_key=KEY)
        assert deepcopy(env) is env

    def test_sandbox_deepcopy_returns_same_instance(self) -> None:
        sandbox = Kosa8Sandbox(client=FakeCloud().client(), create_options={})
        assert deepcopy(sandbox) is sandbox

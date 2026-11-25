# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for Kosa8Sandbox against a fake kosa8 Cloud API."""

import asyncio
import base64
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ag2.annotations import Variable
from ag2.extensions.kosa8.sandbox import Kosa8APIError, Kosa8Sandbox
from ag2.tools.sandbox import ExecResult

from .fake_cloud import FakeCloud, error_answer, exec_answer, out_frame, stream_answer

IN_WORKDIR = ["sh", "-c", 'cd -- "$1" || exit 1; shift; exec "$@"', "ag2"]


def _sandbox(cloud: FakeCloud, **kwargs: Any) -> Kosa8Sandbox:
    options = {"create_options": {"template": "python", "ttl_s": 3600}, **kwargs}
    return Kosa8Sandbox(client=cloud.client(), **options)


class TestConstruction:
    def test_invalid_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            _sandbox(FakeCloud(), timeout=0)

    def test_relative_workdir_rejected(self) -> None:
        with pytest.raises(ValueError, match="absolute"):
            _sandbox(FakeCloud(), workdir="work")

    def test_workdir_is_posix(self) -> None:
        assert _sandbox(FakeCloud(), workdir="/srv").workdir == PurePosixPath("/srv")

    def test_host_workdir_none(self) -> None:
        assert _sandbox(FakeCloud()).host_workdir is None

    def test_invalid_default_env_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="environment variable name"):
            _sandbox(FakeCloud(), env={"1BAD": "x"})

    def test_variable_rejected_in_constructor(self) -> None:
        with pytest.raises(TypeError):
            Kosa8Sandbox(client=Variable("client"), create_options={})  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestExec:
    async def test_maps_argv_environment_workdir_timeout_and_output(self) -> None:
        cloud = FakeCloud(exec_results=[exec_answer("out\n", "err\n", 3)])
        sandbox = _sandbox(cloud, env={"BASE": "1"})

        actual = await sandbox.exec(["python", "-c", "print(42)"], env={"FOO": "bar"}, timeout=12)

        # `ExecResult.output` is contracted to arrive already trimmed.
        assert actual == ExecResult(output="out\nerr", exit_code=3)
        assert cloud.exec_bodies()[-1] == {
            "cmd": [*IN_WORKDIR, "/workspace", "python", "-c", "print(42)"],
            "timeout_s": 12,
            "env": ["BASE=1", "FOO=bar"],
        }
        assert cloud.calls("POST", "/exec")[-1].url.path == "/v1/sandboxes/sbx_1/exec"
        await sandbox.aclose()

    async def test_per_call_env_overrides_sandbox_env(self) -> None:
        cloud = FakeCloud()
        sandbox = _sandbox(cloud, env={"A": "sandbox"})
        await sandbox.exec(["env"], env={"A": "call"})
        assert cloud.exec_bodies()[-1]["env"] == ["A=call"]
        await sandbox.aclose()

    async def test_default_timeout_rounds_up_and_no_env(self) -> None:
        cloud = FakeCloud()
        sandbox = _sandbox(cloud, timeout=7.2)
        await sandbox.exec(["true"])
        assert cloud.exec_bodies()[-1] == {"cmd": [*IN_WORKDIR, "/workspace", "true"], "timeout_s": 8}
        await sandbox.aclose()

    async def test_empty_argv_returns_failure_without_a_sandbox(self) -> None:
        cloud = FakeCloud()
        assert await _sandbox(cloud).exec([]) == ExecResult(output="", exit_code=2)
        assert cloud.created == 0

    async def test_invalid_env_name_rejected_before_anything_runs(self) -> None:
        cloud = FakeCloud()
        with pytest.raises(ValueError, match="environment variable name"):
            await _sandbox(cloud).exec(["env"], env={"A; rm -rf ~; B": "x"})
        assert cloud.created == 0

    @pytest.mark.parametrize(("label", "exit_code"), [("not found", 127), ("not executable", 126), ("killed", 137)])
    async def test_shell_exit_codes_pass_through(self, label: str, exit_code: int) -> None:
        sandbox = _sandbox(FakeCloud(exec_results=[exec_answer(stderr="x", exit_code=exit_code)]))
        assert (await sandbox.exec(["sh"])).exit_code == exit_code, label
        await sandbox.aclose()

    async def test_missing_exit_status_is_a_failure(self) -> None:
        sandbox = _sandbox(FakeCloud(exec_results=[exec_answer("partial", exit_code=None)]))
        actual = await sandbox.exec(["true"])
        assert actual == ExecResult(output="partial\nkosa8 execution ended without an exit status", exit_code=1)
        await sandbox.aclose()

    async def test_server_timeout_returns_124_with_partial_output(self) -> None:
        timed_out = error_answer(504, "exec_timeout", "ran longer than its timeout_s", stdout="tick\n", stderr="")
        sandbox = _sandbox(FakeCloud(exec_results=[timed_out]))
        actual = await sandbox.exec(["sleep", "10"], timeout=3)
        assert actual == ExecResult(output="tick\nkosa8 execution timed out after 3s", exit_code=124)
        await sandbox.aclose()

    async def test_other_api_errors_become_a_failed_result(self) -> None:
        sandbox = _sandbox(FakeCloud(exec_results=[error_answer(409, "not_running", "the sandbox is stopped")]))
        assert await sandbox.exec(["true"]) == ExecResult(output="kosa8 error: the sandbox is stopped", exit_code=1)
        await sandbox.aclose()

    async def test_silent_success_stays_silent(self) -> None:
        sandbox = _sandbox(FakeCloud())
        assert await sandbox.exec(["true"]) == ExecResult(output="", exit_code=0)
        await sandbox.aclose()

    async def test_long_timeout_streams_the_command(self) -> None:
        frames = stream_answer(
            out_frame("a\n"), {"stream": "stdout", "data": ""}, out_frame("b\n", "stderr"), {"done": True, "exit": 4}
        )
        cloud = FakeCloud(exec_results=[frames])
        sandbox = _sandbox(cloud)

        actual = await sandbox.exec(["make"], timeout=600)

        assert actual == ExecResult(output="a\nb", exit_code=4)
        assert cloud.exec_bodies()[-1] == {"cmd": [*IN_WORKDIR, "/workspace", "make"], "timeout_s": 600, "stream": True}
        await sandbox.aclose()

    async def test_streamed_timeout_returns_124(self) -> None:
        frames = stream_answer(out_frame("partial"), {"done": True, "error": "timed out", "code": "exec_timeout"})
        sandbox = _sandbox(FakeCloud(exec_results=[frames]))
        actual = await sandbox.exec(["sleep", "999"], timeout=120)
        assert actual == ExecResult(output="partial\nkosa8 execution timed out after 120s", exit_code=124)
        await sandbox.aclose()

    async def test_streamed_error_is_a_failure(self) -> None:
        frames = stream_answer({"done": True, "error": "container exited"})
        sandbox = _sandbox(FakeCloud(exec_results=[frames]))
        assert await sandbox.exec(["x"], timeout=120) == ExecResult(output="kosa8 error: container exited", exit_code=1)
        await sandbox.aclose()

    async def test_streamed_refusal_is_a_failure(self) -> None:
        refused = error_answer(501, "not_available", "streamed exec is not available")
        sandbox = _sandbox(FakeCloud(exec_results=[refused]))
        actual = await sandbox.exec(["x"], timeout=120)
        assert actual == ExecResult(output="kosa8 error: streamed exec is not available", exit_code=1)
        await sandbox.aclose()

    async def test_host_without_streamed_timeout_gets_the_sandbox_timeout(self) -> None:
        refused = error_answer(501, "not_available", "timeout_s on a streamed exec is not available on this host")
        cloud = FakeCloud(
            exec_results=[
                refused,
                stream_answer(out_frame("ok\n"), {"done": True, "exit": 0}),
                stream_answer(out_frame("again\n"), {"done": True, "exit": 0}),
            ]
        )
        sandbox = _sandbox(cloud)

        assert await sandbox.exec(["make"], timeout=600) == ExecResult(output="ok", exit_code=0)
        assert await sandbox.exec(["make"], timeout=600) == ExecResult(output="again", exit_code=0)

        bodies = cloud.exec_bodies()
        assert bodies[0] == {"cmd": [*IN_WORKDIR, "/workspace", "make"], "timeout_s": 600, "stream": True}
        # The refusal is remembered: later commands go straight to the fallback.
        for body in bodies[1:]:
            assert "timeout_s" not in body
            assert body["stream"] is True
            assert body["cmd"][:2] == ["sh", "-c"]
            assert "timeout" in body["cmd"][2]
            assert body["cmd"][3:] == ["ag2", "/workspace", "600", "make"]
        await sandbox.aclose()

    async def test_sandbox_timeout_kill_returns_124(self, monkeypatch: pytest.MonkeyPatch) -> None:
        refused = error_answer(501, "not_available", "timeout_s on a streamed exec is not available on this host")
        killed = stream_answer(out_frame("partial"), {"done": True, "exit": 143})
        sandbox = _sandbox(FakeCloud(exec_results=[refused, killed]))
        await sandbox.__aenter__()
        clock = iter([0.0, 121.0])
        # The module's own clock only: asyncio keeps the real one.
        monkeypatch.setattr("ag2.extensions.kosa8.sandbox.time", SimpleNamespace(monotonic=lambda: next(clock)))

        actual = await sandbox.exec(["sleep", "999"], timeout=120)

        monkeypatch.undo()
        assert actual == ExecResult(output="partial\nkosa8 execution timed out after 120s", exit_code=124)
        await sandbox.aclose()

    async def test_early_143_is_the_commands_own_exit(self) -> None:
        refused = error_answer(501, "not_available", "timeout_s on a streamed exec is not available on this host")
        exited = stream_answer({"done": True, "exit": 143})
        sandbox = _sandbox(FakeCloud(exec_results=[refused, exited]))
        assert await sandbox.exec(["x"], timeout=120) == ExecResult(output="", exit_code=143)
        await sandbox.aclose()


@pytest.mark.asyncio
class TestFiles:
    async def test_put_file_writes_under_workdir_and_makes_parents(self) -> None:
        cloud = FakeCloud()
        sandbox = _sandbox(cloud)

        await sandbox.put_file(PurePosixPath("pkg/app.py"), b"print(1)\n")
        await sandbox.put_file(PurePosixPath("top.txt"), b"x")

        assert cloud.files == {"/workspace/pkg/app.py": b"print(1)\n", "/workspace/top.txt": b"x"}
        assert cloud.dirs == ["/workspace/pkg"]
        assert cloud.calls("PUT", "/files")[0].headers["Content-Type"] == "application/octet-stream"
        await sandbox.aclose()

    async def test_put_file_falls_back_to_exec_without_the_files_api(self) -> None:
        cloud = FakeCloud(files_error=error_answer(501, "not_available", "no data plane"))
        sandbox = _sandbox(cloud)
        content = bytes(range(256)) * 50  # 12,800 bytes: three chunks

        await sandbox.put_file(PurePosixPath("data/blob.bin"), content)

        writes = [body["cmd"] for body in cloud.exec_bodies()]
        assert len(writes) == 3
        assert all(cmd[5:7] == ["sh", "-c"] for cmd in writes)
        assert "mkdir -p" in writes[0][7] and '> "$1"' in writes[0][7]
        assert all('>> "$1"' in cmd[7] for cmd in writes[1:])
        assert all(cmd[9] == "/workspace/data/blob.bin" for cmd in writes)
        assert b"".join(base64.b64decode(cmd[10]) for cmd in writes) == content
        # Later writes go straight to exec.
        await sandbox.put_file(PurePosixPath("b.txt"), b"")
        assert len(cloud.calls("POST", "/files/mkdir")) == 1
        assert cloud.calls("PUT", "/files") == []
        assert base64.b64decode(cloud.exec_bodies()[-1]["cmd"][10]) == b""
        await sandbox.aclose()

    async def test_put_file_outside_workspace_uses_exec(self) -> None:
        cloud = FakeCloud()
        sandbox = _sandbox(cloud, workdir="/srv")
        await sandbox.put_file(PurePosixPath("a.txt"), b"hi")
        assert cloud.calls("PUT", "/files") == []
        assert cloud.exec_bodies()[-1]["cmd"][9] == "/srv/a.txt"
        await sandbox.aclose()

    async def test_failed_exec_write_raises(self) -> None:
        cloud = FakeCloud(exec_results=[exec_answer(), exec_answer(stderr="read-only", exit_code=1)])
        sandbox = _sandbox(cloud, workdir="/srv")  # the first exec makes the workdir
        with pytest.raises(RuntimeError, match="read-only"):
            await sandbox.put_file(PurePosixPath("a.txt"), b"hi")
        await sandbox.aclose()

    async def test_other_files_api_errors_raise(self) -> None:
        cloud = FakeCloud(files_error=error_answer(413, "too_large", "too big"))
        sandbox = _sandbox(cloud)
        with pytest.raises(Kosa8APIError, match="too big"):
            await sandbox.put_file(PurePosixPath("a/b.txt"), b"x")
        await sandbox.aclose()

    async def test_remove_file_runs_rm_f_in_workdir(self) -> None:
        cloud = FakeCloud()
        sandbox = _sandbox(cloud)
        await sandbox.remove_file(PurePosixPath("old.txt"))
        assert cloud.exec_bodies()[-1]["cmd"][5:] == ["rm", "-f", "/workspace/old.txt"]
        await sandbox.aclose()

    async def test_absolute_paths_rejected(self) -> None:
        sandbox = _sandbox(FakeCloud())
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.put_file(PurePosixPath("/etc/passwd"), b"x")
        with pytest.raises(ValueError, match="Absolute"):
            await sandbox.remove_file(PurePosixPath("/etc/passwd"))


@pytest.mark.asyncio
class TestLifecycle:
    async def test_scope_creates_and_deletes_on_exit(self) -> None:
        cloud = FakeCloud()
        async with _sandbox(cloud, labels={"created-by": "ag2"}, egress={"internet": False}) as sandbox:
            assert sandbox.sandbox_id == "sbx_1"
        assert cloud.created == 1
        assert cloud.deleted == ["sbx_1"]
        assert sandbox.sandbox_id is None
        assert len(cloud.calls("PUT", "/labels")) == 1
        assert len(cloud.calls("PUT", "/egress")) == 1
        # /workspace always exists, so no mkdir exec.
        assert cloud.exec_bodies() == []

    async def test_create_body_and_custom_workdir(self) -> None:
        cloud = FakeCloud()
        async with _sandbox(cloud, workdir="/srv/app"):
            pass
        assert cloud.calls("POST", "/v1/sandboxes")[0].content == b'{"template":"python","ttl_s":3600}'
        assert cloud.exec_bodies() == [{"cmd": ["mkdir", "-p", "/srv/app"]}]

    async def test_sandbox_is_created_once_across_calls(self) -> None:
        cloud = FakeCloud()
        sandbox = _sandbox(cloud)
        await asyncio.gather(*(sandbox.exec(["true"]) for _ in range(5)))
        assert cloud.created == 1
        await sandbox.aclose()

    async def test_waits_for_a_cold_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ag2.extensions.kosa8.sandbox._POLL_INTERVAL_S", 0)
        cloud = FakeCloud(create_state="creating", states_after_create=["creating", "running"])
        async with _sandbox(cloud):
            pass
        assert len(cloud.calls("GET", "/sbx_1")) == 2

    async def test_failed_boot_deletes_the_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ag2.extensions.kosa8.sandbox._POLL_INTERVAL_S", 0)
        cloud = FakeCloud(create_state="creating", states_after_create=["error"])
        sandbox = _sandbox(cloud)
        with pytest.raises(RuntimeError, match="did not start: boom"):
            await sandbox.exec(["true"])
        assert cloud.deleted == ["sbx_1"]
        await sandbox.aclose()

    async def test_setup_error_deletes_the_sandbox(self) -> None:
        cloud = FakeCloud(labels_error=error_answer(400, "invalid_labels", "bad label"))
        sandbox = _sandbox(cloud, labels={"a": "b"})
        with pytest.raises(Kosa8APIError, match="bad label"):
            await sandbox.exec(["true"])
        assert cloud.deleted == ["sbx_1"]
        await sandbox.aclose()

    async def test_workdir_failure_deletes_the_sandbox(self) -> None:
        cloud = FakeCloud(exec_results=[exec_answer(stderr="denied", exit_code=1)])
        sandbox = _sandbox(cloud, workdir="/root/x")
        with pytest.raises(RuntimeError, match="denied"):
            await sandbox.exec(["true"])
        assert cloud.deleted == ["sbx_1"]
        await sandbox.aclose()

    async def test_create_error_propagates(self) -> None:
        cloud = FakeCloud(create_error=error_answer(402, "payment_required", "add a card"))
        sandbox = _sandbox(cloud)
        with pytest.raises(Kosa8APIError) as raised:
            await sandbox.exec(["true"])
        assert (raised.value.status, raised.value.code) == (402, "payment_required")
        await sandbox.aclose()

    async def test_use_after_close_raises(self) -> None:
        sandbox = _sandbox(FakeCloud())
        await sandbox.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.exec(["true"])

    async def test_failed_close_is_not_raised_and_stays_retryable(self) -> None:
        cloud = FakeCloud(delete_errors=[error_answer(502, "host_failed")])
        sandbox = _sandbox(cloud)
        await sandbox.exec(["true"])

        await sandbox.aclose()
        assert sandbox.sandbox_id == "sbx_1"
        await sandbox.aclose()
        assert sandbox.sandbox_id is None
        assert cloud.deleted == ["sbx_1"]

    async def test_sandbox_already_gone_counts_as_deleted(self) -> None:
        cloud = FakeCloud(delete_errors=[error_answer(404, "not_found")])
        sandbox = _sandbox(cloud)
        await sandbox.exec(["true"])
        await sandbox.aclose()
        assert sandbox.sandbox_id is None

    async def test_no_new_sandbox_while_a_failed_one_is_still_alive(self) -> None:
        cloud = FakeCloud(
            labels_error=error_answer(400, "invalid_labels"),
            delete_errors=[error_answer(502, "host_failed"), error_answer(502, "host_failed")],
        )
        sandbox = _sandbox(cloud, labels={"a": "b"})
        with pytest.raises(Kosa8APIError):
            await sandbox.exec(["true"])
        with pytest.raises(RuntimeError, match="could not be deleted"):
            await sandbox.exec(["true"])
        assert cloud.created == 1
        await sandbox.aclose()
        assert cloud.deleted == ["sbx_1"]

    async def test_cancellation_deletes_created_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ag2.extensions.kosa8.sandbox._POLL_INTERVAL_S", 0.01)
        cloud = FakeCloud(create_state="creating", states_after_create=["creating"] * 1000)
        sandbox = _sandbox(cloud)
        task = asyncio.create_task(sandbox.exec(["true"]))
        await cloud.polled.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cloud.deleted == ["sbx_1"]
        await sandbox.aclose()

    async def test_atexit_fallback_deletes_a_sandbox_left_behind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cloud = FakeCloud(delete_errors=[error_answer(502, "host_failed")])
        sandbox = _sandbox(cloud)
        await sandbox.exec(["true"])
        await sandbox.aclose()
        deleted: list[str] = []

        def fake_delete(self: httpx.Client, url: str) -> httpx.Response:
            deleted.append(url)
            return httpx.Response(200)

        monkeypatch.setattr(httpx.Client, "delete", fake_delete)
        sandbox._atexit_close()
        assert deleted == ["/v1/sandboxes/sbx_1"]

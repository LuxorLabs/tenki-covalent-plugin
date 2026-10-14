"""
Unit tests for TenkiExecutor. The tenki SDK is fully mocked -- no
network, no credentials, no real sandboxes.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import cloudpickle
import pytest

from covalent_tenki_plugin.tenki import (
    BOOTSTRAP_COMMAND_TEMPLATE,
    FUNCTION_FILE,
    RESULT_FILE,
    VENV_PATH,
    TenkiExecutor,
)

TASK_METADATA = {"dispatch_id": "d-123", "node_id": 7}


def _version_stdout() -> bytes:
    return f"{sys.version_info[0]} {sys.version_info[1]}\n".encode()


def make_mock_sandbox(result_value=None, result_exception=None, remote_minor=None):
    """Mock sandbox whose exec() responds per-command like the real one."""
    sandbox = MagicMock()
    sandbox.id = "sbx-cov"
    writes = {}

    def exec_side_effect(*argv, **kw):
        joined = " ".join(str(a) for a in argv)
        response = MagicMock()
        response.exit_code = 0
        if "sys.version_info" in joined:
            if remote_minor is not None:
                response.stdout = f"3 {remote_minor}\n".encode()
            else:
                response.stdout = _version_stdout()
        elif joined.startswith("cat ") and RESULT_FILE in joined:
            response.stdout = cloudpickle.dumps((result_value, result_exception))
        elif "cat >" in joined:
            writes[joined] = kw.get("input")
            response.stdout = b""
        else:
            response.stdout = b""
        return response

    sandbox.exec.side_effect = exec_side_effect
    sandbox._writes = writes
    return sandbox


def make_executor(sandbox) -> TenkiExecutor:
    executor = TenkiExecutor(workspace_id="ws-cov")
    client = MagicMock()
    client.create.return_value = sandbox
    executor._make_client = MagicMock(return_value=client)
    executor._client_mock = client
    return executor


class TestRun:
    async def test_round_trip_result(self):
        sandbox = make_mock_sandbox(result_value=42)
        executor = make_executor(sandbox)

        def add(x, y):
            return x + y

        result = await executor.run(add, [40], {"y": 2}, TASK_METADATA)

        assert result == 42
        # the pickled task actually made it into the sandbox
        fn_write = next(v for k, v in sandbox._writes.items() if FUNCTION_FILE in k)
        fn, args, kwargs = cloudpickle.loads(fn_write)
        assert fn(*args, **kwargs) == 42

        # sandbox created with self-destruct + covalent metadata, then cleaned up
        create_kwargs = executor._client_mock.create.call_args.kwargs
        assert create_kwargs["workspace_id"] == "ws-cov"
        assert create_kwargs["max_duration"] == 3600
        assert create_kwargs["metadata"]["covalent-dispatch-id"] == "d-123"
        sandbox.terminate.assert_called_once()
        executor._client_mock.close.assert_called_once()

    async def test_remote_exception_is_reraised(self):
        sandbox = make_mock_sandbox(result_exception=ValueError("boom in vm"))
        executor = make_executor(sandbox)

        with pytest.raises(ValueError, match="boom in vm"):
            await executor.run(lambda: None, [], {}, TASK_METADATA)

        sandbox.terminate.assert_called_once()

    async def test_sandbox_terminated_when_step_fails(self):
        sandbox = make_mock_sandbox()
        sandbox.exec.side_effect = RuntimeError("data plane exploded")
        executor = make_executor(sandbox)

        with pytest.raises(RuntimeError, match="data plane exploded"):
            await executor.run(lambda: None, [], {}, TASK_METADATA)

        sandbox.terminate.assert_called_once()
        executor._client_mock.close.assert_called_once()

    async def test_python_version_mismatch_raises(self):
        wrong_minor = sys.version_info[1] + 1
        sandbox = make_mock_sandbox(result_value=1, remote_minor=wrong_minor)
        executor = make_executor(sandbox)

        with pytest.raises(RuntimeError, match="Python version mismatch"):
            await executor.run(lambda: None, [], {}, TASK_METADATA)

        sandbox.terminate.assert_called_once()

    async def test_custom_image_skips_bootstrap(self):
        sandbox = make_mock_sandbox(result_value=1)
        executor = make_executor(sandbox)
        executor.image = "registry/covalent-ready:1"

        await executor.run(lambda: 1, [], {}, TASK_METADATA)

        joined_calls = [
            " ".join(str(a) for a in call.args) for call in sandbox.exec.call_args_list
        ]
        assert not any("apt-get" in c for c in joined_calls)
        assert (
            executor._client_mock.create.call_args.kwargs["image"]
            == "registry/covalent-ready:1"
        )


class TestConfiguration:
    def test_requirements_always_include_cloudpickle_and_are_quoted(self):
        import covalent

        executor = TenkiExecutor(sandbox_requirements="numpy>=1.26 pandas")
        resolved = executor._resolved_requirements()
        assert resolved.split()[0] == "cloudpickle"
        # covalent must be present in the sandbox to unpickle server-dispatched
        # wrapper callables; pinned to the dispatcher's version
        assert f"covalent=={covalent.__version__}" in resolved
        assert "'numpy>=1.26'" in resolved
        assert "pandas" in resolved

    def test_bootstrap_template_installs_into_venv(self):
        command = BOOTSTRAP_COMMAND_TEMPLATE.format(requirements="cloudpickle")
        assert f"{VENV_PATH}/bin/pip install" in command
        assert "python3 -m venv" in command

    def test_executor_instance_is_cloudpickle_serializable(self):
        # Covalent serializes executor instances; ours must survive it.
        executor = TenkiExecutor(workspace_id="w", cpu_cores=4)
        clone = cloudpickle.loads(cloudpickle.dumps(executor))
        assert clone.workspace_id == "w"
        assert clone.cpu_cores == 4


class TestWorkspaceResolution:
    def test_explicit_workspace_id_wins(self):
        executor = TenkiExecutor(workspace_id="explicit")
        client = MagicMock()
        assert executor._resolve_workspace_id(client) == "explicit"
        client.who_am_i.assert_not_called()

    def test_falls_back_to_first_identity_workspace(self):
        executor = TenkiExecutor()
        client = MagicMock()
        client.who_am_i.return_value = SimpleNamespace(
            workspaces=[SimpleNamespace(id="ws-first"), SimpleNamespace(id="ws-second")]
        )
        assert executor._resolve_workspace_id(client) == "ws-first"

    def test_no_workspaces_raises(self):
        executor = TenkiExecutor()
        client = MagicMock()
        client.who_am_i.return_value = SimpleNamespace(workspaces=[])
        with pytest.raises(RuntimeError, match="No Tenki workspace"):
            executor._resolve_workspace_id(client)

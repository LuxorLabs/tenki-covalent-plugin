"""
Covalent executor plugin that runs each electron (task) inside a disposable
Tenki Sandbox microVM (Firecracker).

Lifecycle per task:

1. Create a sandbox via the official ``tenki`` SDK. ``max_duration``
   acts as a server-side self-destruct so a crashed dispatcher cannot leak
   billed VMs.
2. Bootstrap a Python virtualenv in the sandbox and install ``cloudpickle``
   plus any user-specified requirements (skipped when a prebuilt ``image``
   is given).
3. Ship the cloudpickled ``(function, args, kwargs)`` into the sandbox, run
   it with the sandbox venv Python, and read the pickled
   ``(result, exception)`` back out. No SSH and no object storage needed:
   payloads travel over the SDK's exec data plane.
4. Terminate the sandbox unconditionally.

Authentication uses the ``TENKI_API_KEY`` (or ``TENKI_AUTH_TOKEN``)
environment variable on the machine running the Covalent dispatcher, as
resolved by the SDK. SDK 1.x accepts only ``tk_`` API keys.

Because tasks are transported with cloudpickle, the local Python
interpreter's minor version must match the sandbox's Python (3.12 on the
default image). The executor checks this at bootstrap and fails with a
clear message rather than producing opaque unpickling errors.
"""

from __future__ import annotations

# Defined before the covalent imports: covalent's plugin scanner can observe
# this module mid-import (entry-point load -> covalent.executor.base -> scanner
# re-entry) and requires EXECUTOR_PLUGIN_NAME to already exist at that point.
EXECUTOR_PLUGIN_NAME = "TenkiExecutor"

import asyncio
import shlex
import sys
from typing import Any, Callable, Dict, List, Optional

import cloudpickle

import covalent
from covalent._shared_files import logger
from covalent.executor.base import AsyncBaseExecutor

from tenki import Client

app_log = logger.app_log

_EXECUTOR_PLUGIN_DEFAULTS = {
    "workspace_id": "",
    "cpu_cores": 0,
    "memory_mb": 0,
    "disk_size_gb": 0,
    "image": "",
    "sandbox_max_duration_seconds": 3600,
    "bootstrap_timeout_seconds": 600,
    "task_timeout_seconds": 1800,
    "sandbox_requirements": "",
}

#: Virtualenv inside the sandbox. Ubuntu 24.04 marks the system Python
#: externally managed (PEP 668), so everything installs into a venv.
VENV_PATH = "/home/tenki/.covalent-venv"
WORKDIR = "/home/tenki/covalent-workdir"
FUNCTION_FILE = f"{WORKDIR}/function.pkl"
RESULT_FILE = f"{WORKDIR}/result.pkl"
SCRIPT_FILE = f"{WORKDIR}/exec.py"

BOOTSTRAP_COMMAND_TEMPLATE = (
    "sudo apt-get update -qq"
    " && sudo apt-get install -y -qq python3-pip python3-venv >/dev/null"
    f" && python3 -m venv {VENV_PATH}"
    f" && {VENV_PATH}/bin/pip install --quiet {{requirements}}"
)

#: Remote execution script. Mirrors the approach of Covalent's official
#: remote executor plugins: load the pickled task, run it in the workdir,
#: pickle ``(result, exception)`` to a known path. The unpickle happens
#: inside the try so load-time failures (e.g. a missing import) come back
#: as pickled exceptions instead of an opaque nonzero exit.
EXEC_SCRIPT = f"""
import os
import cloudpickle

result = None
exception = None
os.makedirs({WORKDIR!r}, exist_ok=True)
os.chdir({WORKDIR!r})
try:
    with open({FUNCTION_FILE!r}, "rb") as f_in:
        fn, args, kwargs = cloudpickle.load(f_in)
    result = fn(*args, **kwargs)
except Exception as e:
    exception = e

with open({RESULT_FILE!r}, "wb") as f_out:
    cloudpickle.dump((result, exception), f_out)
"""


class TenkiExecutor(AsyncBaseExecutor):
    """
    Covalent executor that runs each electron in a fresh Tenki Sandbox microVM.

    Args:
        workspace_id: Tenki workspace ID to create sandboxes in. Defaults to
            the first workspace visible to the configured API key.
        cpu_cores: CPU cores for the sandbox (Tenki default if falsy).
        memory_mb: Memory in MB for the sandbox (Tenki default if falsy).
        disk_size_gb: Ephemeral root disk in GB (Tenki default if falsy).
        image: Optional Tenki registry image ref. A prebuilt image with
            cloudpickle preinstalled at ``/home/tenki/.covalent-venv`` skips
            the ~80 s bootstrap.
        sandbox_max_duration_seconds: Hard server-side sandbox lifetime
            (self-destruct safety net). Must exceed bootstrap + task runtime.
        bootstrap_timeout_seconds: Timeout for the apt + pip bootstrap step.
        task_timeout_seconds: Timeout for the task execution itself.
        sandbox_requirements: Extra pip requirement specifiers installed at
            bootstrap (space-separated). ``cloudpickle`` is always installed.
    """

    def __init__(
        self,
        workspace_id: str = "",
        cpu_cores: int = 0,
        memory_mb: int = 0,
        disk_size_gb: int = 0,
        image: str = "",
        sandbox_max_duration_seconds: int = 3600,
        bootstrap_timeout_seconds: int = 600,
        task_timeout_seconds: int = 1800,
        sandbox_requirements: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.workspace_id = workspace_id
        self.cpu_cores = cpu_cores
        self.memory_mb = memory_mb
        self.disk_size_gb = disk_size_gb
        self.image = image
        self.sandbox_max_duration_seconds = sandbox_max_duration_seconds
        self.bootstrap_timeout_seconds = bootstrap_timeout_seconds
        self.task_timeout_seconds = task_timeout_seconds
        self.sandbox_requirements = sandbox_requirements

    # -- helpers (sync, thread-executed) --------------------------------------

    def _make_client(self) -> Client:
        """Create a Tenki SDK client. A new client per task keeps the executor
        instance cloudpickle-serializable (no live gRPC channels on self)."""
        return Client()

    def _resolve_workspace_id(self, client: Client) -> str:
        if self.workspace_id:
            return self.workspace_id
        identity = client.who_am_i()
        if identity.workspaces:
            return identity.workspaces[0].id
        raise RuntimeError(
            "No Tenki workspace visible to this API key; pass workspace_id to "
            "TenkiExecutor explicitly."
        )

    def _resolved_requirements(self) -> str:
        # Tasks dispatched through the Covalent server arrive as covalent
        # wrapper callables, so unpickling them in the sandbox requires
        # covalent itself (same requirement as the official SSH plugin's
        # remote hosts). Pin to the dispatcher's version.
        parts = [
            "cloudpickle",
            f"covalent=={covalent.__version__}",
        ] + (self.sandbox_requirements or "").split()
        return " ".join(shlex.quote(part) for part in parts)

    def _check_python_version(self, sandbox: Any) -> None:
        """cloudpickle payloads require matching interpreter minor versions."""
        probe = sandbox.exec(
            "bash",
            "-lc",
            f'{VENV_PATH}/bin/python -c "import sys; print(sys.version_info[0], sys.version_info[1])"',
            check=True,
        )
        remote = tuple(int(x) for x in probe.stdout.decode().split())
        local = (sys.version_info[0], sys.version_info[1])
        if remote != local:
            raise RuntimeError(
                f"Python version mismatch: local {local[0]}.{local[1]} vs "
                f"sandbox {remote[0]}.{remote[1]}. cloudpickle requires "
                "matching minor versions -- run the dispatcher on "
                f"Python {remote[0]}.{remote[1]}, or supply an image whose "
                f"{VENV_PATH} Python matches your local version."
            )

    def _run_in_sandbox(
        self,
        function_blob: bytes,
        task_metadata: Dict,
    ) -> bytes:
        """Create sandbox, bootstrap, execute the pickled task, return the
        pickled (result, exception) blob. Always terminates the sandbox."""
        dispatch_id = task_metadata.get("dispatch_id", "adhoc")
        node_id = task_metadata.get("node_id", 0)

        client = self._make_client()
        try:
            sandbox = client.create(
                name=f"covalent-{dispatch_id}-{node_id}",
                workspace_id=self._resolve_workspace_id(client),
                cpu_cores=self.cpu_cores or None,
                memory_mb=self.memory_mb or None,
                disk_size_gb=self.disk_size_gb or None,
                image=self.image or None,
                max_duration=self.sandbox_max_duration_seconds,
                tags=["covalent"],
                metadata={
                    "covalent-dispatch-id": str(dispatch_id),
                    "covalent-node-id": str(node_id),
                },
                wait=True,
            )
        except Exception:
            client.close()
            raise

        try:
            app_log.debug("Created Tenki sandbox %s for node %s", sandbox.id, node_id)

            if not self.image:
                bootstrap = BOOTSTRAP_COMMAND_TEMPLATE.format(
                    requirements=self._resolved_requirements()
                )
                sandbox.exec(
                    "bash",
                    "-lc",
                    bootstrap,
                    timeout=self.bootstrap_timeout_seconds,
                    check=True,
                )

            self._check_python_version(sandbox)

            sandbox.exec("mkdir", "-p", WORKDIR, check=True)
            sandbox.exec(
                "bash", "-c", f"cat > {FUNCTION_FILE}", input=function_blob, check=True
            )
            sandbox.exec(
                "bash", "-c", f"cat > {SCRIPT_FILE}", input=EXEC_SCRIPT, check=True
            )
            run_result = sandbox.exec(
                f"{VENV_PATH}/bin/python",
                SCRIPT_FILE,
                timeout=self.task_timeout_seconds,
            )
            if run_result.exit_code != 0:
                stderr_tail = (run_result.stderr or b"").decode(errors="replace")[-2000:]
                raise RuntimeError(
                    f"Task script failed in sandbox {sandbox.id} "
                    f"(exit {run_result.exit_code}):\n{stderr_tail}"
                )

            result_blob = sandbox.exec("cat", RESULT_FILE, check=True).stdout
            return result_blob
        finally:
            try:
                sandbox.terminate()
                app_log.debug("Terminated Tenki sandbox %s", sandbox.id)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                app_log.warning(
                    "Failed to terminate sandbox %s: %s -- max_duration "
                    "self-destruct will reap it.",
                    sandbox.id,
                    exc,
                )
            client.close()

    # -- Covalent executor contract -------------------------------------------

    async def run(
        self,
        function: Callable,
        args: List,
        kwargs: Dict,
        task_metadata: Dict,
    ) -> Any:
        """Run ``function(*args, **kwargs)`` inside a fresh Tenki sandbox."""
        function_blob = cloudpickle.dumps((function, args, kwargs))

        # The tenki SDK is synchronous (gRPC); run in a thread so the
        # dispatcher's event loop keeps servicing other electrons.
        result_blob = await asyncio.to_thread(
            self._run_in_sandbox, function_blob, task_metadata
        )

        result, exception = cloudpickle.loads(result_blob)
        if exception is not None:
            raise exception
        return result

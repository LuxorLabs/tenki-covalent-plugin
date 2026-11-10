# covalent-tenki-plugin

A [Covalent](https://github.com/AgnostiqHQ/covalent) executor plugin that runs
each electron (task) inside a disposable [Tenki Sandbox](https://tenki.cloud)
microVM (Firecracker).

Every task gets a fresh, fully isolated Linux VM with real root — created on
demand, destroyed the moment the task finishes. Ideal for untrusted or
dependency-conflicting workloads: one task's environment can never leak into
another's.

## Installation

```bash
pip install covalent-tenki-plugin
```

`covalent-luxor-ai-plugin` is the same plugin under the Luxor AI name (same
`covalent_tenki_plugin` module and `tenki` executor). Install one or the other,
never both.

## Prerequisites

1. A [Tenki Cloud](https://app.tenki.cloud) account with sandbox access.
2. A Tenki API key, exposed as `LUXORAI_API_KEY` or `TENKI_API_KEY` (or `LUXORAI_AUTH_TOKEN` /
   `TENKI_AUTH_TOKEN`) on the machine running the Covalent dispatcher
   (`LUXORAI_*` and `lx_` keys need `tenki` 1.4.2 or later).
3. Local Python minor version matching the sandbox Python (3.12 on the default
   image) — tasks travel via `cloudpickle`, which requires matching
   interpreters. The executor verifies this and fails fast with a clear
   message on mismatch.

## Usage

```python
import covalent as ct
from covalent_tenki_plugin import TenkiExecutor

executor = TenkiExecutor(
    cpu_cores=2,
    memory_mb=4096,
    sandbox_requirements="numpy pandas",  # installed in the VM at bootstrap
)

@ct.electron(executor=executor)
def process(x):
    import numpy as np
    return float(np.sqrt(x))

@ct.lattice
def workflow(x):
    return process(x)

dispatch_id = ct.dispatch(workflow)(1764)
```

## Configuration

| Argument | Default | Description |
|---|---|---|
| `project_id` | first project on the API key | Tenki project to create sandboxes in |
| `cpu_cores` | Tenki default (2) | Sandbox CPU cores |
| `memory_mb` | Tenki default (4096) | Sandbox memory |
| `disk_size_gb` | Tenki default (5) | Ephemeral root disk |
| `image` | unset (Ubuntu base) | Registry image ref; a prepared image skips the bootstrap |
| `sandbox_max_duration_seconds` | 3600 | Hard VM lifetime (self-destruct safety net) |
| `bootstrap_timeout_seconds` | 600 | Timeout for the apt + pip bootstrap |
| `task_timeout_seconds` | 1800 | Timeout for the task execution |
| `sandbox_requirements` | `""` | Extra pip specs installed at bootstrap (`cloudpickle` and `covalent` always included) |

## How it works

For each electron the executor:

1. **Creates** a sandbox via the official `tenki-sandbox` Python SDK, with
   `max_duration` set so the VM self-destructs server-side even if the
   dispatcher crashes (no leaked billing).
2. **Bootstraps** a virtualenv and installs `cloudpickle`, `covalent` (pinned
   to the dispatcher's version — server-dispatched tasks arrive as covalent
   wrapper callables, so unpickling them requires covalent in the sandbox,
   same as the official SSH plugin's remote hosts), and your
   `sandbox_requirements` (~2–3 min on the default image; skipped when
   `image` points at a prepared registry image).
3. **Ships** the cloudpickled `(function, args, kwargs)` into the VM over the
   SDK's exec data plane (no SSH, no object storage), executes it with the
   sandbox venv Python, and reads the pickled `(result, exception)` back.
4. **Terminates** the sandbox unconditionally; remote exceptions are re-raised
   locally with full fidelity.

## Current limitations

- **~2–3 min cold start** per task on the default image (apt + venv +
  `covalent` install) — significant for many short electrons. Use a prepared
  `image` to eliminate it, or batch small steps into fewer electrons.
- Task payloads and results travel through the exec data plane; very large
  results (tens of MB) may hit message-size limits — write large artifacts to
  external storage from within the task instead.
- Mid-task cancellation is not yet wired to Covalent's cancel API; the
  `max_duration` self-destruct bounds runaway tasks.
- Sandbox volumes and snapshots are not used by this plugin.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q          # unit tests, fully mocked, no credentials needed
```

A live end-to-end smoke test requires `TENKI_API_KEY` and creates (then
terminates) real sandboxes.

### Known upstream issue (unrelated to this plugin)

With `covalent==0.240.0` and `requests>=2.34`, **any** `ct.dispatch()` fails
with `422 Unprocessable Entity`: the SDK posts the dispatch manifest as a raw
string without a `Content-Type: application/json` header, and the server's
FastAPI rejects it. Workaround until fixed upstream:

```python
from covalent._dispatcher_plugins import local as ldisp

_orig_post = ldisp.APIClient.post
def _patched_post(self, endpoint, **kw):
    if "data" in kw:
        kw.setdefault("headers", {})["Content-Type"] = "application/json"
    return _orig_post(self, endpoint, **kw)
ldisp.APIClient.post = _patched_post
```

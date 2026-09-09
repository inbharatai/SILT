# Experimental Studio: Compose and Compiler

An additive, opt-in local workbench for the real public `python -m asea.compose`
and `python -m asea.compiler` commands. It does not call the legacy Studio
catalog or share loaded model weights with it. Importing the router does not
import Torch/Transformers. No fixture backend exists in this UI.

**A successful Compose run is not admission or activation. All Compiler outputs
remain UNADMITTED.** The workbench installs nothing and downloads no models.
See [COMPOSITION_V1.md](COMPOSITION_V1.md) and [COMPILER_V1.md](COMPILER_V1.md)
for supported architectures, input schemas, runtime dependencies and limitations.

## Opt-in launch and integration contract

The server must mount the exported router **only** under the exact feature flag:

```python
if os.environ.get("SILT_ENABLE_EXPERIMENTAL") == "1":
    from .experimental import router as experimental_router
    app.include_router(experimental_router)
```

The router independently returns a typed `DISABLED` 404 unless the flag is `1`.
It creates no workspace or worker at import. It creates them lazily when an
experimental route is first used. Legacy endpoints and legacy `.studio` remain
separate. Use one server process/worker, bound to loopback:

```sh
cd /agent/workspace/silt-local
SILT_ENABLE_EXPERIMENTAL=1 \
SILT_EXPERIMENTAL_ROOT=/agent/workspace/silt-experiments \
PYTHONPATH=src \
/agent/workspace/silt-venv/bin/python -m uvicorn asea.studio.server:app \
  --host 127.0.0.1 --port 8377 --workers 1
```

Use your installed Python/virtualenv path if different. The example paths are
local operator choices, not required defaults. Open
`http://127.0.0.1:8377/experimental` on the same machine. The page has no CDN or
external JavaScript dependencies. Leave any existing legacy Studio dependency
checks disabled through the server's own documented configuration when using a
lightweight environment; the experimental router itself never probes/imports
ML dependencies. This document does not invent a separate dependency-disable flag.

`SILT_EXPERIMENTAL_ROOT` defaults to `.experimental-studio`, resolved relative to
the server's working directory. Choose a dedicated directory, never your source
checkout, home directory or filesystem root. The root is made owner-private
(mode `0700`). Paths inside any legacy `.studio` directory and symlink roots are
rejected. A POSIX file lock prevents a second worker using the same root.
**POSIX is required** for process-group cancellation; unsupported platforms fail
closed rather than offering unreliable cancellation. Serial execution is per
experimental root; running another root, old Studio jobs or unrelated local
programs simultaneously can still consume memory. This is not an OS memory
sandbox or a guarantee against OOM.

## Public UI workflow

1. Choose **Compose** or **Compiler** and an operation.
2. Enter your operator name and applicable local spec/suite/model/calibration
   paths, text prompt, input PCM WAV or a PNG/JPEG/WebP image. These are server-machine paths, not
   browser uploads. Absolute paths are recommended; relative input paths resolve
   from the server/CLI working directory, not the spec directory.
3. Confirm that you authorize reading the files and any paths referenced inside
   their JSON. This is a trusted single-user tool: names are audit declarations,
   not authenticated user accounts.
4. Click **Queue real CLI job**. Watch queued/running/terminal statuses and select
   a job to see exact argv, raw JSON, exit code and stderr. Full stdout/stderr are
   shown after the process finishes, not fictional progress percentages.
5. Use **Cancel process group** to cancel a running job or remove a queued job
   from execution. Cancelled queued jobs can occupy a pending slot briefly until
   the worker drains them. Capture/save raw job JSON for your own evidence.
6. Download only listed produced artifacts. Artifact IDs are not arbitrary file
   paths. Compiler candidate files have a distinct `compiler_candidate` role,
   evaluation JSON `compiler_evidence`, Compose WAV `compose_audio`, and deployment
   ZIP `deployment_export`. The browser uses streaming save when its File System
   Access API is available. Other browsers refuse >100 MiB buffered downloads;
   use the authenticated curl approach below for large files.

Output input is deliberately a **single filename/directory name**, not a free
absolute output path: each job gets `<root>/jobs/<job-id>/artifacts/<name>`.
Defaults are `candidate`, `evaluation.json`, and `deployment.zip`. Spaces, slashes,
leading hyphens and traversal are rejected. Produced output must not exist.
This confines writes without limiting trusted local model input locations.

### Operations and exact CLI translation

The executable is always `sys.executable`; modules and operations come from a
fixed allowlist. No shell, executable field, arbitrary command or extra flags
are accepted. Values use `--flag=value`, including prompt text beginning with
`-`, so they cannot become flags.

| Workflow / operation | Form → public CLI |
|---|---|
| Compose inspect | `--workspace <root>/compose inspect --spec=PATH` |
| Compose plan | `--workspace <root>/compose plan --spec=PATH` (schema-only draft, no quality/admission claim) |
| Compose run | `--workspace <root>/compose run --spec=PATH [--input=TEXT] [--input-file=PCM_WAV_OR_IMAGE]` |
| Compose evaluate | `--workspace <root>/compose evaluate --spec=PATH --suite=PATH` |
| Compose select | `--workspace <root>/compose select --evaluations ID ... --input-type=text/audio/image --output-type=text/audio --max-wall-seconds=N --max-peak-rss-mb=N [--suite-hash=SHA256] [--max-disk-bytes=N] [--graph-hash=SHA256]` |
| Compose activate | `--workspace <root>/compose activate --evaluation=ID [--approve-high-risk --approver=NAME]` |
| Compose rollback | `--workspace <root>/compose rollback --deployment=ID` |
| Compose export | `--workspace <root>/compose export --deployment=ID --output=CONTAINED_PATH [--include-dependencies]` |
| Compose list | `--workspace <root>/compose list` |
| Compiler inspect | `inspect --model=PATH` |
| Compiler prune | `prune --model=PATH --output=CONTAINED_PATH --keep-experts=N --calibration=PATH` plus runtime options |
| Compiler evaluate | `evaluate --model=PATH --suite=PATH [--reference-model=PATH] --evidence-output=CONTAINED_PATH` plus runtime/generation options |
| Compiler infer | `infer --model=PATH --prompt=TEXT` plus runtime/generation options |

Compose text graphs accept text only; audio graphs accept a PCM WAV file only.
Image graphs require a bounded PNG/JPEG/WebP file and may include a text question
(default: `Describe the image.`). Evaluation cases use `input_file` plus optional
`input` for image questions. Both fields together are rejected for text/audio
graphs before inference. Input evidence binds image bytes and the question;
activation rechecks this digest, the current runtime version and the graph hash.
No vision model quality is claimed by the UI.

Select accepts at most 64 bounded evaluation IDs (whitespace-separated in the
form, an array in the API), never arbitrary flags. Without a suite hash it
returns `no_feasible_bundle`; matching I/O alone is not capability evidence.
Plan and select never activate and do not load models.

Compiler runtime options are explicit `--dtype={float32,float16}` (default
float32), `--max-length=1..256` (128), `--memory-budget-mib=128..3072` (2048).
Generation uses `--max-new-tokens=1..128` (32). Inspect does not receive runtime
flags. Repair training is unsupported by the public CLI and is deliberately not
exposed as a UI action. The compiler's own preflight checks still apply; a budget
is not permission to exceed detected RAM.

### Activation/rollback are never automatic

High-risk activation requires a named `--approver=NAME`; the boolean
`--approve-high-risk` alone is insufficient. The UI forwards both flags **only** for explicit `compose / activate`, after a named
operator has entered `ACTIVATE` and checked the separate high-risk approval box.
The operator name, confirmation and boolean approval are persisted in Studio job
evidence. Without the checkbox no override flag is sent: the underlying CLI
must reject high-risk activation if approval is required. Underlying certification
continues checking admission, hashes and provenance. The checkbox does not bypass
those checks. Studio does not expose a different activation implementation.
Rollback similarly requires the exact word `ROLLBACK` and a named operator.
Neither changes the legacy Studio active pointer.

## API contract / automation schema

All experimental APIs, including job reads, schema and artifact downloads,
require `X-SILT-Experimental-Token`. The random per-server-session token is
embedded in `/experimental` only after local-origin checks. Reload the page when
the server restarts. Tokens are not put into download URLs, query parameters or
logs. All responses use `Cache-Control: no-store`.

| Method | Route | Result |
|---|---|---|
| GET | `/experimental` | Local UI with session token and CSP nonce |
| GET | `/api/experimental/schema` | Exact Pydantic request JSON Schema, actions and limits |
| POST | `/api/experimental/jobs` | 202 `{ok:true,job:{...}}`; strict JSON request |
| GET | `/api/experimental/jobs` | Bounded newest-first job summaries |
| GET | `/api/experimental/jobs/{id}` | Raw stdout, stderr, parsed result, argv, status, error, artifact manifest |
| POST | `/api/experimental/jobs/{id}/cancel` | Current job; cancellation request is idempotent |
| GET | `/api/experimental/jobs/{id}/artifacts/{artifact_id}` | Contained produced regular file, streaming `FileResponse` attachment |

Example request (no model needed):

```json
{
  "mode": "compose",
  "operation": "list",
  "operator": "Local operator",
  "local_files_confirmed": true,
  "timeout_seconds": 300
}
```

Example real failure-path request: set `mode: "compiler"`, `operation: "inspect"`,
`model: "/absolute/nonexistent/model"`, plus operator and local permission fields.
This invokes the actual public CLI and shows its typed failure; it is not a
successful model test.

For a large artifact, an authorized local operator can copy the token from the
page's source/devtools and keep it in a temporary shell variable (do not share
or record it):

```sh
curl --fail -H "X-SILT-Experimental-Token: $SILT_TOKEN" \
  http://127.0.0.1:8377/api/experimental/jobs/JOB_ID/artifacts/ARTIFACT_ID \
  --output candidate-file.safetensors
unset SILT_TOKEN
```

Bridge errors are `{ok:false,error:{type,message}}` with appropriate HTTP codes.
Known types include `AUTH_REQUIRED`, `ORIGIN_REJECTED`, `LOCAL_ONLY`,
`INVALID_REQUEST`, `REQUEST_TOO_LARGE`, `INVALID_ACTION`, `MISSING_ARGUMENT`,
`CONFIRM_LOCAL_FILES`, `CONFIRM_ACTIVATION`, `CONFIRM_ROLLBACK`, `UNSAFE_PATH`,
`UNSAFE_OUTPUT`, `NOT_OWNER`, `OUTPUT_EXISTS`, `QUEUE_FULL`, `WORKSPACE_BUSY`,
`JOB_NOT_FOUND`, and `ARTIFACT_NOT_FOUND`. Once a job is accepted, execution errors
are in the job record: `CLI_FAILED`, `INVALID_CLI_JSON`, `SPAWN_ERROR`,
`WORKER_ERROR`, `TIMEOUT`, `OUTPUT_LIMIT`, `CANCELLED`, `SERVER_RESTARTED`.
The underlying CLI's own error object is preserved under `job.result.error`.
A nonzero CLI exit or `ok:false` is never reported as a successful job.

## Execution, persistence and bounds

- One subprocess at a time; up to **three waiting**; queue-full returns 429.
- Child process gets offline HF/Transformers settings, two OpenMP/MKL threads,
  disconnected stdin, separate process session and no shell. ML dependencies are
  imported only by the public CLI when its operation actually requires them.
- Timeout default **300 seconds**, accepted **1..900**. Cancel, timeout and output
  flood signal the whole process group (TERM then KILL after 0.5 s). Descendants
  are also killed when the parent CLI finishes. This assumes trusted CLI code
  does not deliberately escape its group; it is not a hostile-code sandbox.
- Combined captured stdout + stderr capped at **10 MiB**. On overflow the process
  group is stopped; only bounded capture is retained. Request bodies are capped
  at 128 KiB; prompt 32768 characters. JSON escaping and parsed-evidence duplication
  can make the persisted evidence file larger than the captured byte count.
- Latest **20 job records** retained in memory and on disk under
  `jobs/<id>/evidence.json` with a bounded `stderr.log`. Writes use atomic JSON
  replacement, owner-only evidence/log permissions and a private root. Stdout,
  raw stderr, parsed JSON, CLI argv, return code, operator approval and state are
  retained. Restart marks interrupted records failed and **never replays** them.
- Retention deletes old evidence/logs, **not user-produced artifacts or Compose
  workspace records/models**. Export/candidate files can be large and require
  operator-managed disk cleanup offline. Expired-job artifacts are no longer
  downloadable through this API. Keep/export evidence before it ages out.
- Graceful server shutdown cancels the running process group. A SIGKILL of the
  entire server cannot run cleanup; inspect orphan processes before restarting.
  Normal restarts do not replay work; the root lock controls active server
  managers, not arbitrary orphaned processes or independent roots.

## Trust and browser security boundary

This is **one trusted local OS operator**, not multiple remote tenants. Bind only
to `127.0.0.1`; do not reverse-proxy/expose it on a LAN or public network. Operator
names are not verified identities. Any local process with the operator's OS
privileges can acquire the UI token or read their files.

The router checks a loopback hostname (protecting against arbitrary Host/DNS
rebinding), rejects cross-origin `Origin` and cross-site Fetch Metadata even on
the token-bearing UI route, and checks the token on **every API read and write**.
This is necessary because legacy Studio may allow selected public/LAN CORS
origins: those origins still cannot read this token page or call these APIs.
It adds no permissive CORS policy. UI framing is forbidden by CSP; scripts/styles
use nonces and remote dependencies are absent. Evidence is rendered with
`textContent`, never interpreted as HTML.

Explicit existing input files/directories must be owned by the current OS user
and not symlinks/special files. Missing paths reach the public CLI for honest
validation errors. JSON specs/suites may reference other local inputs: the
operator explicitly authorizes these, and the public CLI's own path/model safety
checks apply. This is not general filesystem confinement of model inputs. There
is no browse/open-file endpoint and no arbitrary path download endpoint.
Downloads require both a recorded produced role and resolved containment beneath
the experimental root; symlinks are rejected, files are streamed as attachments.
A hostile same-UID process racing file replacement is outside this trust model;
`FileResponse` containment is not a substitute for OS isolation.

## Tests and orchestrator GUI acceptance

```sh
PYTHONPATH=src /agent/workspace/silt-venv/bin/python -m pytest \
  tests/test_experimental_studio.py tests/test_integration_contracts.py -q
```

Tests cover auth on reads/writes/downloads, disabled gate, inherited-origin
rejection, exact argv/no flag injection, named high-risk confirmation, path
containment, serial queue, running/queued cancellation and descendant cleanup,
timeout, output flood, malformed JSON/nonzero errors, evidence retention/restart,
produced artifact downloads and import without Torch. Orchestration tests use
test-only subprocess doubles. A real compiler inspect subprocess against a
missing model checks the authentic CLI failure path. These tests establish no
model quality and do not replace real browser acceptance. Integration contracts
also import the actual legacy server in fresh processes with the flag unset,
`0`, `true`, and `1`; compare legacy routes and static/README bytes; verify every
experimental API independently rejects missing/wrong tokens and public origins;
and preserve real Compose CLI errors through the new endpoint. Image binding
checks use adapter doubles, not downloaded vision weights, and verify that
changed image bytes, questions, run digests or runtime versions block activation.

For user-level browser automation by the orchestrator:

1. Enable flag and mount router, start loopback server with `PYTHONPATH=src`.
2. Open `/experimental`; assert heading, form and loaded workspace appear.
3. Choose Compose → list, enter operator, check local permission, click Queue.
   Verify real succeeded job and raw Compose `ok:true` JSON, no autoactivation.
4. Choose Compiler → inspect, enter an intentionally missing local model path,
   queue; verify failed status, nonzero exit, raw typed CLI failure.
5. Use an actual supported provisioned model/spec for real inference only if the
   operator has budgeted memory. No production fixture switch or fake adapter.
6. Verify mode/operation-dependent controls, evidence-save, responsive layout,
   cancel for an actual running operation, and authenticated produced download
   when a real artifact is available. Record each outcome honestly; do not
   imply model/download/cancel acceptance when only unit tests exercised it.

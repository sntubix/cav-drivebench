# DriveBench

DriveBench is an assessed course project in which student teams build the
decision system around an unreliable vision-language model (VLM) driving a
simulated vehicle. MetaDrive supplies the world and vehicle simulation;
DriveBench supplies the local control, safety, model-provider, evaluation, and
replay boundaries around it.

MetaDrive `0.4.3` requires Python `>=3.6,<3.12`, so this project supports Python
`>=3.10,<3.12`.

## How the course works

Your team writes code only in `submission/`, never in files already in the
tree. Each assignment adds one piece of the decision system around the model,
and every file ships as a working stub, so the whole pipeline drives from the
first day:

| Assignment | You write | It decides |
| --- | --- | --- |
| 1. PID control | `controller.py` | steering, throttle, and brake from tracking errors |
| 2. Observation and arbitration | `observation.py`, `arbitration.py` | what the model sees, and how far to trust its assessment |
| 3. Request cadence and cloud | `request_policy.py` | when to ask the model, within a request budget |
| 4. Race | no new code | tuning only |

`submission/agent.yaml` holds the tuning values your assignment permits. Run
your submission by passing its directory, and check it against the structural
gates that grading also runs, on hidden scenarios. The same command scores it:

```bash
uv run metadrive-starter smoke --submission submission
uv run metadrive-starter gates
```

Everything else, including the simulator, route tracking, the safety floor,
model providers, and evaluation, is instructor-owned foundation that your code
calls but never edits. Run the public scenarios as often as you like; graded
numbers come only from hidden scenarios that the instructors run. The rules
your files must follow are in [Your submission](docs/getting-started.md#your-submission).
Start with the [Assignment 1 brief](course/assignment-1/README.md).

## Local Setup

Run all commands from the repository root unless stated otherwise.

The course publishes DriveBench as a git repository. Your team works in a
private repository of its own and pulls each release from the course's, kept as
the `upstream` remote: [Get the code and stay up to date](docs/getting-started.md#get-the-code-and-stay-up-to-date)
sets that up in a few commands, and shows how to pull a release.

Install [`uv`](https://docs.astral.sh/uv/) if needed, then create the project
environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group dev
```

Verify the wiring, then launch one real headless simulation:

```bash
uv run metadrive-starter smoke --dry-run --submission submission
uv run metadrive-starter run \
  --config configs/demo-autopilot.yaml \
  --submission submission \
  --headless \
  --steps 1000
```

The smoke check validates configuration, planning, perception, and control
without opening MetaDrive. The second command launches the deterministic
autopilot profile and should complete its route without a crash or off-road exit.
The controller in `submission/` drives both, and a student release has no other.

For rendered, manual, autopilot, and emergency-braking demonstrations, continue
with the [getting-started guide](docs/getting-started.md).

## Student learning path

1. [Install, run, and configure DriveBench](docs/getting-started.md).
2. [Run, check, and tune your submission](docs/getting-started.md#your-submission),
   starting with [Assignment 1](course/assignment-1/README.md).
3. [Understand the VLA pipeline and local authority boundaries](docs/vla-pipeline.md).
4. [Connect the supported local or cloud model provider](docs/model-providers.md).
5. [Probe model behaviour before granting vehicle authority](docs/evaluation.md#open-loop-vla-probe).
6. [Evaluate closed-loop behaviour and replay decisions](docs/evaluation.md#seeded-closed-loop-evaluation-and-scheduled-faults).

All assignments in the published course sequence are required. Instructors
provide the cloud accounts used by the course, and the supplied profiles enforce
request caps. Students do not provision cloud projects or administer billing.

## Docker

Linux is a supported project platform. Docker is the reproducible Linux
headless baseline and is the path exercised by the automated platform validator:

```bash
docker compose build
docker compose run --rm metadrive \
  metadrive-starter run --submission submission --headless --steps 100
```

On a Linux host, evaluate the complete automated Docker slice with:

```bash
./scripts/platform-check.sh
```

The validator covers image construction, the frozen dependency lock, bundled
MetaDrive assets, tests, a real headless route, offscreen camera capture, the
fixture-backed VLA loop, and deterministic replay. It writes immutable evidence
under `artifacts/platform-validation/<UTC timestamp>/` and explicitly records
manual or live-model gates that were not run.

The same local `uv` workflow targets Linux and macOS. Native Ubuntu VLM
qualification uses the pinned [`native-platform-check.sh` recipe](docs/model-providers.md#linux-local-server-acceptance-llamacpp);
it verifies the native simulator path and two clean local-model runs. Use the
Docker validator for the reproducible container baseline.

## Configuration

Defaults live in `configs/default.yaml`, and the starting PID gains every profile
drives with live only in `configs/pid.yaml`. Select another profile with `--config`:

```bash
uv run metadrive-starter run \
  --config configs/demo-autopilot.yaml \
  --submission submission
```

Configuration owns simulator settings, controller gains, local planning,
perception, safety, provider selection, model request policy, event logging, and
fault injection. Runtime/domain speeds are expressed in m/s; rendered displays
may opt into km/h. See [Getting started: Configuration](docs/getting-started.md#configuration).

Profiles choose the world you experiment in. The values your team is graded on
belong in `submission/agent.yaml`, which applies on top of whichever profile
you run and may set only the keys your assignment permits.

## VLA assessment and command validation

The VLM is an advisor, never the actuator. It returns a descriptive assessment;
local code derives a time-bounded command, validates it against the local scene,
and only then permits it to influence planning. The instructor-owned safety
floor remains final authority.

See [VLA pipeline and authority](docs/vla-pipeline.md) for the schema, validation
dispositions, scheduling, action-speed policy, lane-change checks, and safety
layers.

## Camera and model-input boundary

Camera capture is disabled for ordinary PID/manual runs. VLA profiles enable a
front RGB camera and send its lossless image together with the deterministic
prompt and bounded local-scene context. Model servers remain isolated behind a
single provider contract.

See [Camera and observation](docs/vla-pipeline.md#camera-and-model-input-boundary).

## Model providers and inference scheduling

DriveBench supports deterministic fixtures, OpenAI-compatible local HTTP
servers, and the course Vertex provider through the same pipeline. Course
support applies only to the documented engine/model profiles:

- MLX-VLM on Apple Silicon;
- `llama.cpp`/`llama-server` on Linux;
- Vertex for the instructor-funded cloud stage.

Other compatible engines may work, but are not supported until their exact
runtime/model combination passes the project acceptance sequence. See
[Model providers](docs/model-providers.md) for setup and credential loading.

## In-process model option

The default architecture keeps heavyweight model runtimes outside the project
environment. Researchers may inject an in-process provider factory when an
experiment genuinely needs it; configuration never imports arbitrary Python
classes. See [In-process providers](docs/model-providers.md#in-process-model-option).

## Open-loop VLA probe

The probe command captures a fixed visual catalog without granting vehicle
authority. It can then send those exact images and prompts to a provider, score
the returned assessments, or replay saved inputs against another provider.

Start with:

```bash
uv run metadrive-starter probe --output-dir tmp/vla-probe-run-01
```

See [Open-loop VLA probe](docs/evaluation.md#open-loop-vla-probe) for scenarios,
artifacts, live inference, exact-input replay, and lane-choice comparisons.

### Lane-choice provider comparison

Capture lane-choice inputs once and replay the exact bytes across providers. See
[Lane-choice provider comparison](docs/evaluation.md#exact-input-replay).

## Seeded closed-loop evaluation and scheduled faults

`evaluate` expands deterministic scenario/seed matrices, records every episode,
and can inject camera, LiDAR, provider, latency, and malformed-output faults at
fixed simulation steps. See [Seeded closed-loop evaluation](docs/evaluation.md#seeded-closed-loop-evaluation-and-scheduled-faults).

### Deterministic lane-change acceptance suite

Use the public fixture-backed suite to verify lane admission, rejection, and
abort behaviour without model nondeterminism. See
[Deterministic lane-change acceptance](docs/evaluation.md#deterministic-lane-change-acceptance).

## Reproducible public and hidden race suites

Students may repeatedly run the public race suite while tuning. Graded results
come only from an instructor-controlled hidden evaluation set; its commands,
manifests, and raw artifacts are not part of the student documentation or
release.

See [Race evaluation](docs/evaluation.md#reproducible-public-and-hidden-race-suites)
for the public workflow and evidence contract.

## Event logging and decision replay

Append-only JSON Lines events distinguish model requests, validated commands,
planner/controller output, safety overrides, and applied controls. `replay`
reruns deterministic local decisions; `race-replay` plays recorded MetaDrive
world states.

See [Event logging and decision replay](docs/evaluation.md#event-logging-and-decision-replay).

## Layout

<details>
<summary>Repository layout</summary>

```text
configs/                         simulator, controller, provider, and evaluation profiles
src/metadrive_starter/           application package and CLI
src/metadrive_starter/controllers/ vehicle-controller protocol and factory
src/metadrive_starter/planning/  route tracking and future-path projection
src/metadrive_starter/perception/ local-scene adapters
src/metadrive_starter/safety/    command validation and emergency safety
src/metadrive_starter/vla/       camera, transport, assessment, and scheduling
scripts/platform-check.sh        Linux Docker validation and evidence capture
scripts/native-platform-check.sh native Ubuntu VLM validation and evidence capture
tests/                           unit, integration, and simulator checks
submission/                      team-owned code, starting from working stubs
course/assignment-1/             Assignment 1 brief, rubric, and PID lab notebook
```

</details>

## Notes

- `metadrive_starter.env` is the only module that imports MetaDrive directly.
- Local safety remains authoritative regardless of the selected model provider.
- Generated evidence belongs under `tmp/` or `artifacts/`; never commit
  credentials, unreviewed model responses, or private hidden-evaluation output.

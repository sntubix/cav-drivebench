# Evaluation and replay

[← Model providers](model-providers.md) · [DriveBench README](../README.md) · [Finish: Learning path ↩](../README.md#student-learning-path)

Run all commands from the repository root unless stated otherwise.

DriveBench separates model judgement, closed-loop behaviour, deterministic local
decisions, and simulator-world playback. Each layer produces its own evidence;
passing one is not evidence that every other layer works.

## Open-loop VLA probe

The model-independent probe captures the fixed visual catalog without contacting
a provider by default:

```bash
uv run metadrive-starter probe --output-dir tmp/vla-probe-run-01
```

The checked manifest contains deterministic clear-road, stopped-vehicle,
curved-road, intersection, red-light-with-traffic, adjacent-traffic,
partial-occlusion, and all-lanes-blocked scenes. Each scenario declares
acceptable tactical actions, expected hazard observations, and a diagnostic
target-speed range. Against a stopped vehicle in the ego lane, the expected
action is a change into a clear adjacent lane, and waiting only when every lane
is blocked; slowing down still closes on it, so it is never accepted.

Every scenario directory contains:

- `frame.png`: the lossless PNG of the observation's image, as sent to providers;
- `prompt.txt`: the observation's exact prompt;
- `scene.json`: the bounded local-scene context measured at capture, which
  DriveBench's own prompt embeds and a team's may not;
- `metadata.json`: placement, dimensions, timing, hashes, request identity,
  parsed assessment/command, rubric layers, and outcome;
- `raw_response.txt`: untouched provider output when inference returns text.

The artifact root also contains `summary.json`. Output directories are created
exclusively and never overwritten.

Capture selected scenes only:

```bash
uv run metadrive-starter probe \
  --scenario stopped-vehicle \
  --scenario partial-occlusion \
  --output-dir tmp/vla-probe-hazards-01
```

After starting a supported server, perform one request per scenario:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-local.yaml \
  --infer \
  --output-dir tmp/vla-probe-model-01
```

Inference remains open-loop: assessments and locally derived commands are saved
but never applied to the vehicle. Transport or parsing failures still produce
metadata and retain raw model text. The command exits non-zero if any selected
scenario fails technically.

Semantic evaluation reports perception, tactical-action, and raw-speed results
separately. Raw speed is diagnostic unless a manifest explicitly makes it part
of the gate, because local action-speed and safety policy own enforced speed.
`summary.json` counts, per layer, the scenes that passed out of those scored.

Probe your own observation by passing your submission. Each scene's observation
is then built by your `observation.py`, after your `agent.yaml`, and saved exactly
as the model receives it:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-local.yaml \
  --submission submission \
  --infer \
  --output-dir tmp/vla-probe-team-01
```

The probe scores the model's assessment, never your arbiter's answer.

### Exact-input replay

Replay a saved catalog against another provider without rerendering MetaDrive:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-local.yaml \
  --infer \
  --replay-from tmp/vla-probe-run-01 \
  --output-dir tmp/vla-probe-replay-01
```

Replay verifies the saved RGB, PNG, prompt, and scene hashes. Artifacts from
before team observations (schema 2 and 3) must also still embed their exact
`LOCAL_SCENE_CONTEXT`. It sends the saved observation's original bytes, whoever
built it, and writes responses to a new directory; the source evidence is never
changed. `--submission` does not combine with `--replay-from`.

<details>
<summary>Lane-choice provider comparison</summary>

Capture the shared lane-choice inputs once:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-lane-change-qwen3-vl-2b-mlx.yaml \
  --manifest configs/vla-probe-lane-change-scenarios.yaml \
  --output-dir tmp/lane-choice-inputs-01
```

Replay those exact bytes through a local provider:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-lane-change-qwen3-vl-2b-mlx.yaml \
  --infer \
  --replay-from tmp/lane-choice-inputs-01 \
  --output-dir tmp/lane-choice-local-01
```

The first frame has a stationary lead and a clear right lane; only
`CHANGE_LANE_RIGHT` is accepted. The second adds a right-lane blocker and accepts
only `FOLLOW`, `SLOW_DOWN`, or `STOP`. Both require the relevant vehicle hazards.
This is evidence of model judgement, never vehicle authority.

</details>

## Seeded closed-loop evaluation and scheduled faults

`evaluate` expands a strict YAML manifest across scenarios and seeds, runs every
episode independently, and writes one event log per run plus an aggregate
`summary.json`:

```bash
uv run metadrive-starter evaluate \
  --config configs/demo-emergency-braking.yaml \
  --manifest configs/evaluation-scenarios.yaml \
  --submission submission \
  --output-dir tmp/evaluation-01
```

Use a new output directory. One failed episode is recorded and does not prevent
later matrix entries from running. `--dry-run` validates the matrix and artifact
layout without launching MetaDrive.

Fault intervals are deterministic and step-based: `start_step` is inclusive and
`start_step + duration_steps` is exclusive. Supported kinds are:

- `camera_dropout`;
- `lidar_dropout`;
- `provider_timeout`;
- `provider_error`;
- `provider_latency` with `latency_s`;
- `malformed_output`.

Activation/clear edges, request fault IDs, provider outcomes, command decisions,
controls, and terminal outcomes remain in the event log. A fault cannot silently
create authority: prior commands remain bounded by their original horizon, your
arbiter's `review` can end them sooner, and the independent emergency supervisor
remains final.

The Assignment 2 structural gates publish one fault scenario per kind in
[`configs/gates-assignment-2.yaml`](../configs/gates-assignment-2.yaml), in the same form, beside the routes. A fault
scenario need not arrive. Its gate, "faults never break the run", fails only on a
crash, an off-road exit, or a run that stopped:

```bash
uv run metadrive-starter gates \
  --manifest configs/gates-assignment-2.yaml \
  --config configs/demo-vla-fixture.yaml
```

Grading uses other timings, maps, and combinations of the same kinds. The run
reports no score, because Assignment 2 is marked from hidden runs as its rubric
says. The manifest's `no_score` text says so in the report.

### Deterministic lane-change acceptance

The public fixture-backed suite tests local validation and execution without
live-model nondeterminism. Student releases include its fixtures from Assignment 2
on:

```bash
uv run metadrive-starter evaluate \
  --config configs/lane-change-fixture.yaml \
  --manifest configs/lane-change-scenarios.yaml \
  --submission submission \
  --output-dir tmp/lane-change-suite-01
```

Its cases cover successful right-lane change, rejection at the left road edge,
rejection for unsafe rear gap/TTC, and abort after a new target-lane obstacle.
Every case also requires no crash and no off-road result. This proves foundation
behaviour, not whether a VLM chooses the correct lane. Your controller steers the
change, so the completion case tests it too: the shipped stub does not reach the
target lane before the 8 s timeout and aborts.

## Reproducible public and hidden race suites

The public race is the student practice and tuning set:

```bash
uv run metadrive-starter race \
  --config configs/demo-autopilot.yaml \
  --manifest configs/race-public-v1.yaml \
  --submission submission \
  --output-dir tmp/race-public-01
```

Each case fixes its map, seed, traffic pattern, obstacle probability, stopped
vehicle, timing, and horizon. Controller, planner, safety, prompt, provider, and
request policy remain student inputs. The scoring path is headless so rendering
does not distort model latency.

Each scenario records a complete MetaDrive episode under
`<output>/<scenario>/replay/`; `summary.json` stores its path and SHA-256. Replay
the exact trajectories later without rerunning the submitted policy:

```bash
uv run metadrive-starter race-replay \
  --race-output tmp/race-public-01 \
  --render \
  --trust-artifact
```

Graded results come only from an instructor-controlled hidden evaluation set
using the same capability classes but different exact worlds. Instructor
commands, manifests, and raw outputs are excluded from student releases. Students
receive only the approved aggregate feedback defined by the course.

Published race versions are immutable. Change cases or locked runtime fields by
creating a new version rather than editing the existing manifest.

## Event logging and decision replay

Enable append-only JSON Lines logging in YAML or per run:

```bash
uv run metadrive-starter run \
  --config configs/demo-emergency-braking.yaml \
  --submission submission \
  --event-log tmp/emergency-run.jsonl
```

The log separates the requested model command, validated effective command,
executor objective, PID proposal, emergency override, and MetaDrive action. It
also records configuration, correlation, latency/failures, scene and ego state,
safety transitions, threats, and terminal outcome. Prompts, raw images, model
responses, API keys, and environment secrets are not embedded.

Replay deterministic validator, executor, headway, lane-clearance/coordinator,
and emergency-safety decisions:

```bash
uv run metadrive-starter replay \
  --event-log tmp/emergency-run.jsonl
```

Decision replay is not bit-exact physics replay. It reconstructs typed inputs and
configuration and reruns the local policy layers in event order. `race-replay`
is separate: it plays MetaDrive's recorded world frames and verifies actor pose
at every recorded control frame.

MetaDrive episode payloads are pickle. Always require `--trust-artifact`, and
never load student-provided or otherwise untrusted replay payloads. A SHA-256 can
detect changed bytes; it cannot make pickle safe.

### Hazard agreement

Score the hazards in every logged assessment against the local scene it was
validated against, per model profile:

```bash
uv run python -m metadrive_starter.hazard_agreement tmp/
```

Pass event logs or directories, which are searched for `*.jsonl`; `--json` prints
the same report as JSON. It reports hazard precision and recall, recall of
hazards in the ego path, how often an assessment agrees with its scene, and
confidence binned against that agreement. Assessments from fixtures and replays
are reported apart from live model calls, never pooled with them. It reads logs
only, and runs neither the simulator nor a model.

<details>
<summary>Visual evidence checklist</summary>

Before accepting a camera/model runtime, inspect exported images beside their
prompts. Images must be upright and non-mirrored, use correct RGB colours, show
useful road and horizon proportions, preserve lane markings and traffic at
relevant distances, and remain semantically comparable across rendered and
headless capture.

Confirm that the server accepts the lossless PNG data URL without an undocumented
resize or crop. Test ordinary decoding before strict structured output so prompt
compliance remains measurable, then run the structured configuration as a
separate comparison.

</details>

## End of the student path

Return to the [README learning path](../README.md#student-learning-path), or use
the public commands above while working through the assignments.

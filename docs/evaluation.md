# Evaluation and replay

[← Model providers](model-providers.md) · [DriveBench README](../README.md)

You need this guide from Assignment 2 on; Assignment 1 uses none of it.

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
  parsed assessment/command, semantic evaluation layers, and outcome;
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
scenario fails technically. Assignment 2's marks come from this probe, run on
hidden scenes with your submission, as its brief's "How you are marked" states;
no mark comes from a closed-loop run.

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

### Probe scores and Assignment 2's marks

`summary.json` scores each layer in `scores`, as scenes passed out of scenes
scored. Each is close to one of [Assignment 2's marks](../course/assignment-2/README.md#the-mark), not the
same:

| Layer | Passes when | Close to |
| --- | --- | --- |
| a scene's `status` | `success` when the answer decoded; `failure`, with a decoder error in `metadata.json`, when it did not, and also when no answer came back: a timeout or transport error, which marking does not count, or a request the server refused, which it counts as an answer that does not decode | [syntax](../course/assignment-2/README.md#syntax-10-points) |
| `perception` | the answer reports every hazard the scene requires | the recall in [scene interpretation](../course/assignment-2/README.md#scene-interpretation-15-points) |
| `tactical_action` | the proposed action is one the scene accepts | [tactical decision](../course/assignment-2/README.md#tactical-decision-30-points) |
| `raw_speed` | the proposed speed is in the scene's range; diagnostic, since the runtime maps each action to its own speed | [tactical decision](../course/assignment-2/README.md#tactical-decision-30-points) |

Scene interpretation is marked per hazard, on scenes of its own.
`scores.perception` passes a scene whose answer reports every hazard it
requires, which is close to recall, but it never checks for invented hazards,
which cost precision, and it requires road features, which marking never
counts. [`probe-analysis.ipynb`](../course/assignment-2/probe-analysis.ipynb) lists each hazard an
answer invents as "not in the scene", and pools a run's answers into recall,
precision, and F1 as marking does. The tactical-decision mark also needs the
speed to fit the scene, before and after your arbiter, which the probe never
runs.

The probe's traffic lights are enlarged to 1.75 times their size, so that their
state is legible at 512x288, and a traffic-light result on the probe does not
generalise to real signals. Assignment 2 marks no traffic light: no hidden
scene holds one, so the public red-light scene is for practice only.

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
changed.

With `--submission`, `--replay-from` asks instead through your `observation.py`:
it needs a recorded catalog, whose scenes keep their measured `local_scene.json`,
such as those in `fixtures/vla/probes/`, and builds each observation from the
recorded frame and scene, without MetaDrive.

<details>
<summary>Lane-choice provider comparison</summary>

Capture the shared lane-choice inputs once. The profile below is an MLX one,
for demos and plumbing checks; its figures do not predict Assignment 2's marks,
as [Model providers](model-providers.md#apple-silicon-local-server-mlx-vlm) says:

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
only `FOLLOW` or `STOP`. Both require the relevant vehicle hazards.
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
- `malformed_output`;
- `provider_quota`, a refusal for quota, as an HTTP 429;
- `provider_outage`, the model unavailable, as an HTTP 503.

Every request that reaches the provider counts against
`vla.maximum_requests_per_run`, whether an injected fault fails it or not.

Activation/clear edges, request fault IDs, provider outcomes, command decisions,
controls, and terminal outcomes remain in the event log. A fault cannot silently
create authority: prior commands remain bounded by their original horizon, your
arbiter's `review` can end them sooner, and the independent emergency supervisor
remains final.

The Assignment 2 structural gates publish one fault scenario per kind in
[`configs/gates-assignment-2.yaml`](../configs/gates-assignment-2.yaml), in the same form, beside the routes. A fault
scenario need not arrive. Its gate, "faults never break the run", fails only on a
run an error stopped; a crash or an off-road exit is how the controller drove, so it
is reported and fails no gate. The gates drive with a recorded answer, so they
never ask a model:

```bash
uv run metadrive-starter gates \
  --manifest configs/gates-assignment-2.yaml \
  --config configs/demo-vla-fixture.yaml
```

Grading uses other timings, maps, and combinations of the same kinds. The run
reports no score, because Assignment 2 is marked from the open-loop probe on
hidden scenes, as its brief says. The manifest's `no_score` text says so in the
report.

### What an arbiter sees under each fault

The public gate run schedules one scenario of each of the six kinds; grading
uses other timings, maps, and combinations, two at once. An Assignment 2
arbiter has only to survive them, and this is what each hands it:

| Fault | What `arbitrate` sees |
| --- | --- |
| `camera_dropout` | no new request, so no assessment; the command that holds authority grows old |
| `lidar_dropout` | with LiDAR as the safety source, a scene with `valid` false and no objects |
| `provider_timeout` | no assessment |
| `provider_error` | no assessment |
| `provider_latency` | an answer that arrives late; `now_s` minus `issued_at_s` is its age |
| `malformed_output` | no assessment |

The three provider failures reach only `review`, as `request.failure`, which
Assignment 2 leaves returning True, so each leaves the last command your
arbiter let through to drive until its action horizon ends. Retrying belongs
to Assignment 3.

The Assignment 3 structural gates, in the manifest its brief names, give
every scenario a `request_budget`, which replaces `vla.maximum_requests_per_run`
for that drive, and fail requests four ways: `provider_timeout`,
`malformed_output`, `provider_quota`, and a long `provider_outage`. One more
gate, "request budget respected", checks every budgeted drive's request
accounting:

```bash
uv run metadrive-starter gates \
  --manifest configs/gates-assignment-3.yaml \
  --config configs/demo-vla-fixture.yaml
```

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

### Comparing observation and arbitration live

Two live drives differ in everything the two submissions differ in, the controller
included: a weak speed gain leaves a car at 0.4 m/s where a tuned one has it at
3 m/s, and that alone changes which proposals arrive in time. To compare
`observation.py` and `arbitration.py` by themselves, drive both with one controller
and one set of driving values. In a checkout with the instructor reference, this
drives a submission with the reference's `controller.py` and tuning:

```bash
uv run metadrive-starter run \
  --config configs/demo-vla-qwen3-vl-2b-llama.yaml \
  --submission submission \
  --reference-driving \
  --headless --realtime-factor 0.25 --seed 1
```

It keeps the rest of your `agent.yaml`, which your observation and arbiter read, and
replaces the keys that tune the driving. Without `--reference-driving`, `--submission`
loads everything from your directory, as it always does. A student release has no
reference, so the option stops there; compare two versions of a seam under your own
controller instead.

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
are reported apart from live model calls, never pooled with them, and each
fixture apart from every other, even when two share a model id. It reads logs
only, and runs neither the simulator nor a model.

### The two recorded failures

[`configs/demo-vla-semantic-stopped-vehicle.yaml`](../configs/demo-vla-semantic-stopped-vehicle.yaml) and
[`configs/demo-vla-semantic-partial-occlusion.yaml`](../configs/demo-vla-semantic-partial-occlusion.yaml) replay the two
recorded wrong answers of
[the Assignment 2 brief](../course/assignment-2/README.md#two-failures-in-opposite-directions). With the
shipped arbiter, the model's command holds authority on 199 of 200 steps in
both, and only the safety floor stops the car, 3.0 m short
(`minimum_safety_gap_m`). Declining hands the same approach to the local
fallback, which the floor stops 3.0 m short too.

Hazard agreement also gives each log's confidence AUROC, the chance that an
agreeing assessment is more confident than a disagreeing one, where 0.5 is
chance: n/a in each, since each log repeats one answer. Pooled, the two would
give 1.00 only because their answers differ in confidence as well as in
agreement, so never pool runs from different sources.

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

## Measuring Assignment 2 work

[The Assignment 2 brief](../course/assignment-2/README.md#measuring-and-citing) asks you to name, for every
figure, the field it was read from and the command that wrote it. The probe's
`metadata.json` and `run`'s JSON summary give every figure at full precision;
`gates` gives none to cite. A live model can answer differently every time, so
keep every output directory you cite; figures from the recorded answers and
scenes reproduce exactly.

| Tool | Covers | Does not cover |
| --- | --- | --- |
| [`observation-inspector.ipynb`](../course/assignment-2/observation-inspector.ipynb) | what your `observation.py` sends on the eight probe scenes and the eight recorded scenes with pedestrians and obstacles; what your file and each `agent.yaml` value change from DriveBench's original; what lower resolutions keep of the smallest things the probe checks | any model's answer |
| `probe --submission ... --infer` | one answer per scene from one model, scored per layer | your arbiter; a closed loop |
| [`probe-analysis.ipynb`](../course/assignment-2/probe-analysis.ipynb) | probe output by criterion, hazard agreement against the measured scene, recall, precision, and F1, SmolVLM's accepted answers, your arbiter on each answer, reruns, event logs; with no runs of your own, the two recorded failures on every scene | anything you did not give it |
| `run ... --event-log` | a closed loop: the summary's `vla_metrics`, and one event per decision and control tick | scores against a scene's expected answer |
| `python -m metadrive_starter.hazard_agreement` | the hazards and confidence of every assessment in event logs | probe output |
| `gates`, Assignment 2's manifest | the structural floor, with a recorded answer | model quality, and your Assignment 2 marks |

The notebooks are never submitted or graded, so save them as you like.

Measure each attempt in a copy of your submission, never by editing and
reverting `submission/` itself:

```bash
mkdir -p tmp && cp -R submission tmp/attempt-1
uv run metadrive-starter probe --config configs/vla-probe-smolvlm-llama.yaml --submission tmp/attempt-1 --infer --seed 1 --output-dir tmp/probe-attempt-1-01
```

Edit `tmp/attempt-1/` between the two commands. `probe`, `run`, `gates`, and
`config` all take `--submission` with any directory, and the probe saves the
exact prompt each attempt sent. Record the attempt's complete `agent.yaml`, or
its exact code change, beside the command, so that anyone can rebuild it.

## Where to go next

Return to the [README](../README.md#start-here), or use the public commands above
while working through the assignments.

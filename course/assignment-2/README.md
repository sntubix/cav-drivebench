# Assignment 2: Observation and arbitration

[← DriveBench README](../../README.md) · [Observation inspector](observation-inspector.ipynb) · [Probe analysis](probe-analysis.ipynb)

> Assignment weight: **30%**

The course calendar gives the deadline, and late submissions are not accepted.
Run all commands from the repository root.

From now on a vision-language model (VLM) assesses each scene and proposes what
the car should do. It is an unreliable advisor, and the system around it is the
engineering. You decide what the model sees, the observation, and how far to
believe what it says back, arbitration. Rewording a prompt answers neither: the
prompt is configuration, and its measured effect is what earns marks. The
[glossary](../../CONTEXT.md) defines the course's terms.

**What Assignment 2 marks.** Your observation and arbiter sit in the live
driving system and shape how the car drives, and the gates drive routes and
fault scenarios with them to check that nothing breaks. The marks come from
still scenes: one frame, one observation, one assessment at a time, each in a
fresh episode, as [How you are marked](#how-you-are-marked) says. How well the
car drives earns nothing, history your arbiter keeps across assessments earns
nothing, and `review` stays as shipped. Drive live to explore your work and to
show it at the oral defence.

## Steps

1. [Set up](#set-up), and check that the shipped files pass Assignment 2's
   gates.
2. [Serve SmolVLM](#serving-the-models), [probe](#probing-your-observation) the
   shipped observation with it, and
   [choose a realtime factor](#choosing-a-realtime-factor) from the latencies
   the probe records.
3. Work through [the task](#the-task) with the two notebooks,
   [measuring](#measuring-and-citing) every change on SmolVLM and confirming
   it on Qwen, which earns most of the marks.
4. [Check your work](#check-your-work) with the gates.
5. [Hand in](#what-you-hand-in) a zip of `submission/` on Moodle.

## Set up

```bash
uv sync --group dev --extra notebooks
uv run metadrive-starter gates --manifest configs/gates-assignment-2.yaml --config configs/demo-vla-fixture.yaml
```

Every gate should report `PASSED`. The run drives the five Assignment 1 routes
and six fault scenarios with a recorded model answer, so it needs no model
server, and it prints no score: Assignment 2 is marked from hidden runs, as
[How you are marked](#how-you-are-marked) says. It takes a few minutes, and
longer the first time in a new environment, such as a new clone or folder,
because MetaDrive downloads its assets once into each.

Then open the observation inspector, which needs neither MetaDrive nor a model:

```bash
uv run --extra notebooks jupyter lab course/assignment-2/observation-inspector.ipynb
```

## The task

Every gap in the shipped [`submission/observation.py`](../../submission/observation.py) and
[`submission/arbitration.py`](../../submission/arbitration.py) is marked by a named `TODO` line, and each
file's docstring states the contract it must keep. The two notebooks take the
`TODO` lines as their tasks, and end each task with the structural gates' own
checks, printed as `PASS` or `NOT YET`:

1. **Your observation**: what the model receives, and where each fact goes:
   in the text, drawn into the image, or left for the model to read from the
   pixels. The [observation inspector](observation-inspector.ipynb) takes its
   `TODO` lines as tasks 1 to 3: the instructions, which hold two, the facts the
   text gives, and the image. Start with [The model ladder](#the-model-ladder):
   its question, and what of SmolVLM is marked.
2. **Your arbiter**: how far to believe each assessment, the model's answer.
   Catch the unsound assessments and keep the sound ones, while only ever
   taking authority, a command's permission to influence the vehicle, away.
   [Probe analysis](probe-analysis.ipynb) takes its `TODO` lines as one task.
   Start from the [two recorded failures](#two-failures-in-opposite-directions).

The model's answers to your observation earn the syntax, SmolVLM, and
scene-interpretation marks, and your arbiter earns the tactical-decision mark.
Grading asks Qwen3-VL-2B for all of them but 5 points, which SmolVLM earns with
answers under the grammar that the runtime accepts, so SmolVLM alone cannot tell
you how your work will be marked: probe your observation on Qwen too, with structured output off and
on, in planned sessions on your team's strongest laptop. Each request names the
model it goes to, so your observation can prompt each model its own way.

Throughout, **keep a record** of what you try and measure, with the command
behind each figure: there is no written report, but at the oral defence your
team explains and analyses its work. Some of these questions have no fix, and
concluding so with evidence is a sound answer. The defence asks, among others:

- why SmolVLM, which follows a one-word instruction, does not follow your
  observation without the grammar ([The model ladder](#the-model-ladder));
- whether your observation makes SmolVLM's commands sound on a scene, and why
  they are unsound on another. No mark depends on how sound SmolVLM's commands
  are;
- whether a model's confidence carries signal your arbiter can use
  ([Your arbiter](#your-arbiter)).

## The model ladder

You work with two models, a weak one first:

| Model | Role | Files, from [Model providers](../../docs/model-providers.md) |
| --- | --- | --- |
| SmolVLM-256M, Q8_0 | the instrument you develop against, which earns 5 points with answers under the grammar that the runtime accepts | `SmolVLM-256M-Instruct-Q8_0.gguf` and its `mmproj` |
| Qwen3-VL-2B, Q4_K_M | the stronger profile, which earns the rest | `Qwen3VL-2B-Instruct-Q4_K_M.gguf` and its `mmproj` |

Each model is two files: the language model, and its `mmproj`, the part that
reads the image; `Q8_0` and `Q4_K_M` name how far its weights are compressed.
A server answers either unconstrained, writing whatever the model likes, or
under JSON Schema output, also called structured output, where a grammar lets
the model write only text that fits the assessment's schema. DriveBench's
decoder then turns the text into an assessment; an answer it refuses does not
decode.

**SmolVLM is a teaching instrument, not a smaller Qwen.** Without a grammar it
answered every probe request with a prose caption of the image, so no answer
decoded, although it follows a one-word instruction. With the grammar every
answer takes the assessment's form, whatever it understood: a cheap, local
source of well-formed, wrong assessments, exactly what arbitration exists to
catch. The grammar holds the form only, not the values in it, so an answer
whose numbers fall outside the schema's ranges is still refused.

Your work on the observation starts from one question, which the oral defence
asks: **why does a model that follows a one-word instruction not follow this
one?** Answer it from your own measurements, by probing without the grammar.
No mark depends on SmolVLM decoding without the grammar, and none is expected
to. What is marked is SmolVLM with the grammar on:
[SmolVLM answers the runtime accepts](#smolvlm-answers-the-runtime-accepts-5-points)
is the share of its structured answers that decode and that the runtime would
act on, to the observation your `observation.py` builds when `request.model_id`
names SmolVLM, and those answers reach your arbiter beside Qwen's. A way in: in
[a copy of your submission](#measuring-and-citing), give SmolVLM a prompt of its
own, probe it with
[`configs/vla-probe-smolvlm-llama-structured.yaml`](../../configs/vla-probe-smolvlm-llama-structured.yaml),
and read what it answered and why each answer was refused or discarded, in
that scene's `metadata.json`; then check that Qwen's answers did not change.

### Serving the models

Serve one model at a time on port 8000 with the pinned `llama-server`, as
[Model providers](../../docs/model-providers.md#serving-a-model-for-assignment-2) describes, then pass
the configuration for that server with `--config`:

| Configuration | Model | Output |
| --- | --- | --- |
| [`configs/vla-probe-smolvlm-llama.yaml`](../../configs/vla-probe-smolvlm-llama.yaml) | SmolVLM | unconstrained: the model answers as it will |
| [`configs/vla-probe-smolvlm-llama-structured.yaml`](../../configs/vla-probe-smolvlm-llama-structured.yaml) | SmolVLM | JSON Schema: the grammar forces the form |
| [`configs/vla-probe-qwen3-vl-2b-llama.yaml`](../../configs/vla-probe-qwen3-vl-2b-llama.yaml) | Qwen | unconstrained |
| [`configs/vla-probe-qwen3-vl-2b-llama-structured.yaml`](../../configs/vla-probe-qwen3-vl-2b-llama-structured.yaml) | Qwen | JSON Schema |
| [`configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml`](../../configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml) | Qwen | JSON Schema, without lane clearance, as scene interpretation asks |
| [`configs/demo-vla-smolvlm-llama.yaml`](../../configs/demo-vla-smolvlm-llama.yaml), [`configs/demo-vla-qwen3-vl-2b-llama.yaml`](../../configs/demo-vla-qwen3-vl-2b-llama.yaml) | either | a short closed-loop run, JSON Schema |

The endpoint, the model, and whether the grammar applies are instructor-owned,
so `agent.yaml` cannot change them. They sample at temperature 0.7, as grading
does, but under one seed the same input gets the same answer, so a rerun repeats
the last unless its seed changes. The one exception is the server's prompt
cache: a first answer can differ from later ones that reuse its cached prompt.
Grading serves with `--no-cache-prompt`, so that each answer depends only on
its request and seed; add it too when a run must repeat exactly, at the cost of
reading every prompt in full. Vary the seed as grading's reruns do: `--seed 1`
on `probe` or `run`, then `--seed 2`, and so on, and report every live figure as
the median and range over those runs.
On Apple silicon, do all marked work on
[the macOS arm64 build](../../docs/model-providers.md#apple-silicon-the-macos-arm64-build-of-the-same-pin)
of the same pinned `llama-server`, on Metal, serving the same files. The
[MLX profiles](../../docs/model-providers.md#apple-silicon-local-server-mlx-vlm)
answer differently, because their checkpoints are not the pinned files, so
observation figures from MLX do not predict your marks; keep them for demos and
plumbing checks. On Linux with a GPU,
[the Vulkan build](../../docs/model-providers.md#gpu-the-vulkan-build-of-the-same-pin)
of the same pinned `llama-server` serves the same files with the same flags,
far faster than on the CPU, and it is the build grading uses.

## Hardware and pacing

Grading runs on instructor hardware, so your laptop changes how you measure,
never your mark. SmolVLM runs on any laptop. Qwen3-VL-2B is about ten times
slower per answer and, without a discrete GPU, can near the 60 s request
timeout, so run it on your team's strongest laptop, together: the server stays
on that machine's loopback. Qwen's server needs about 4 GiB of memory and
SmolVLM's under 1 GiB, beside about 5 GiB for the simulator, whose camera
render slows every closed-loop run.

A closed-loop run earns no mark. Assignment 2 is marked from the open-loop
probe on hidden scenes, as [How you are marked](#how-you-are-marked) says, and
the gates drive with a recorded answer. A live run is for exploring and showing
your work: your observation and arbiter driving with the model answering live,
and your arbiter meeting answers that arrive as late as your model takes. The
run asks the model as often as it answers, one request at a time, the next sent
on the tick the last answer arrives: nothing samples on a timer, and under a
local server nothing caps how many requests a run makes.

### Choosing a realtime factor

While a request is out, simulated time keeps passing at the rate
`--realtime-factor` on `run` sets: at 0.25, four wall-clock seconds pass per
simulated second. Every threshold is in simulated time, so the factor changes
none of them, only how old each answer is when it arrives: a 6 s answer is 6 s
old at 1.0 and 1.5 s old at 0.25. The safety floor refuses an answer that
arrives older than `command_validation.maximum_command_age_s`, 2 s by default.
One it accepts holds authority until its action horizon, counted from its
frame: `vla.action_horizon_s`, 2 s by default and 3 s in the model demos.
Requests go out one at a time, so the next answer lands as long after this one
as this one took. For the model to hold authority without gaps, each answer
must arrive within half the horizon:

1. **Measure the latency**: `outcome.latency_s` in each probe scene's
   `metadata.json`, or `vla_metrics.latency_p95_s` in a run's summary. Take the
   slow end.
2. **Divide half the horizon by it**, up to 1.0. With the default 2 s horizon,
   4 s gives 0.25 and 10 s gives 0.1; under the model demos' 3 s horizon, 6 s
   gives their 0.25.
3. **Check that your machine kept up**: one that cannot runs slower than asked.
   In a run's `--event-log`, the simulated seconds between the first and last
   `control_applied` event, over the wall-clock seconds between their
   `wall_time_utc`, is the factor achieved; an `inference_completed` event's
   `sim_time_s`, minus the `submitted_at_s` in its payload, is the age its
   answer arrived at.

Report the factor, and the achieved one if lower, with every closed-loop
figure: a slow factor makes a slow model look fast, and interpreting that is
part of the assessed work. Grading never slows time: it ages each answer by
its token counts, as [Tactical decision](#tactical-decision-30-points) states,
and so does [`probe-analysis.ipynb`](probe-analysis.ipynb), from the counts each probe
records: test how your arbiter treats an answer's age there. A closed-loop run
gives it each answer as late as your model took.
`--unpaced` runs as fast as the machine allows, and the probe is never paced.

## Your observation

[`submission/observation.py`](../../submission/observation.py) defines `ObservationBuilder(settings)`, and its
docstring states the contract: for every model request, DriveBench calls
`build(request)` and sends the model exactly the `Observation` it returns, one
`RGBFrame` and the prompt text. The request carries the frame, the ego and
cruise speeds, the parts of the measured scene the model may have, and
`model_id`, the model it goes to, so one file can suit each model. Which facts
to include, and at what fidelity, is yours.

**The image is the model's only source of objects.** Assignment 2's requests
withhold the measured objects and traffic lights: `request.withheld` names
them, and `request.scene` holds which lanes beside yours exist, whether each is
observed clear, and the distances to the road's edges, but no object. The model
must find vehicles, pedestrians, and obstacles in the image, and
[scene interpretation](#scene-interpretation-15-points) marks what it finds
there. The hidden interpretation scenes carry lane existence but not clearance,
since whether a lane is clear says on which side an object stands:
`request.withheld` names `lane_clearance` too, and `request.scene` leaves both
lanes' clearance as `None`, so build an observation that works without it. Your
arbiter still receives the whole measured scene.

Two rules hold whatever you build:

- Keep the frame's `timestamp_s`: crop, scale, or draw on the image, but never
  re-date it.
- `build` must return within the gates' request timeout, 1 s under
  [`configs/demo-vla-fixture.yaml`](../../configs/demo-vla-fixture.yaml) and at grading, or the
  ["control loop never blocked"](#check-your-work) gate fails. The probe
  configurations allow 60 s, so a `build` that takes 2 s passes every probe and
  fails the gate. The observation inspector's checks hold it to 1 s.

**The output contract is a recommended default.** `request.contract` holds four
lines that ask for exactly what the decoder accepts. Keep each verbatim on a
line of its own, anywhere in the prompt, or rephrase or replace them: a run
never refuses an observation without them, only warns and counts it in
`vla_metrics.observations_without_output_contract`. But without a grammar, a
model not held to the schema rarely answers in a form the decoder accepts, and
that is what your syntax mark measures.

### The values in `agent.yaml`

Assignment 2 permits every Assignment 1 key and these:

| Keys | What they set | Allowed values |
| --- | --- | --- |
| `vla.prompt_policy` | instructions of your own; the shipped observation places them after `CONFIGURED_POLICY:` | any text |
| `observation.driving_context`, `observation.scene_context` | whether the shipped observation includes the measured driving context and scene context | `true` or `false` |
| `observation.scene_fields.lanes`, `.traffic_controls`, `.objects` | which parts of the scene context it includes; in Assignment 2 only `lanes` changes anything, since the request already withholds objects and traffic controls | `true` or `false` |
| `command_validation.minimum_confidence` | the confidence below which the floor falls back | only raised, at most 1 |
| `command_validation.lane_change_front_gap_m`, `lane_change_rear_gap_m`, `lane_change_minimum_ttc_s` | the clear gap ahead and behind, in m, and the time to collision, in s, a lane change needs | only raised |
| `command_validation.maximum_command_age_s`, `maximum_clock_skew_s`, `scene_stale_after_s` | the oldest a command may be, how far in the future its timestamp may lie, and the oldest the measured scene may be, in s, before the floor falls back | only lowered |
| `command_validation.maximum_target_speed_mps`, `slow_down_speed_mps`, `yield_speed_mps` | the highest speed any command may set, and the highest a `SLOW_DOWN` or a `YIELD` may set, in m/s | only lowered |

Any other key, or a value outside its bounds, is rejected by name. The
observation toggles act only through an `observation.py` that reads them, as
the shipped one does. `command_validation` holds the safety floor's thresholds,
which your arbiter also receives as its `settings`: you may make the floor
stricter, never looser.

<details>
<summary>Why one configuration can reject a value another accepts</summary>

"Only raised" and "only lowered" count from the configuration a run passes
with `--config`. The closed-loop model demos set `controller.target_speed_mps`
and `maximum_target_speed_mps` to 5.56 m/s, where the other configurations set
9.72 and 13.9, so a value above 5.56 is rejected only in the demos. Giving
`config` the same `--config` prints what a run will use, for example
`uv run metadrive-starter config --config configs/vla-probe-smolvlm-llama.yaml --submission submission`.

</details>

### The observation inspector

[`observation-inspector.ipynb`](observation-inspector.ipynb) builds your
observation, with your `agent.yaml`, on the eight public probe scenes, and on
[the eight recorded scenes](#probing-your-observation) with pedestrians, cones,
and barriers. It shows
the image and the text side by side, what your file and each `agent.yaml` value
change from DriveBench's original, and what lower resolutions keep of the
smallest things the probe checks: what the model *receives*, never what it
answers. The notebooks are never submitted or graded, so save them as you
like.

## Your arbiter

[`submission/arbitration.py`](../../submission/arbitration.py) defines `Arbiter(settings)`, with the
`command_validation` thresholds as its `settings`. Its docstring, and
[VLA pipeline](../../docs/vla-pipeline.md#arbitration), state the contract. In short,
`arbitrate(request)` sees each assessment as it arrives and `review(request)`
runs every control tick, both on the control loop, so each must return within
one control step, 0.1 s at the default timing; `begin_episode()` runs before
each episode's first assessment, to clear any history. Assignment 2 asks for
`arbitrate` only: leave `review` returning True, as shipped. The runtime already
ends a command at its action horizon, aborts a lane change into a lane that is
no longer clear, and brakes for what is ahead, and ending a command early hands
control to a fallback that drives at the cruise speed, so in the course's own
runs a `review` that ended commands changed nothing.

**You may only take authority away**, the monotone restriction. `arbitrate` may
lower `proposed_target_speed_mps`, lower `confidence` (below
`settings.minimum_confidence` the floor falls back), or decline by replacing
`proposed_action` with `REQUEST_FALLBACK`; `review` may keep or end the command
that holds authority, never extend or change it. Any other answer, or a call
that raises, is overruled, and the "monotone restriction preserved" gate
fails; an exception from `begin_episode()` stops the run. Declining or ending a
command hands control to the local fallback, which drives at the cruise speed
with the safety floor.

The shipped arbiter endorses everything. The evidence so far:

- **Hazard agreement carries signal.**
  `metadrive_starter.hazard_agreement.score_hazards` scores the hazards an
  assessment reports against `request.scene`, as the course measured them,
  quickly enough for every call.
- **Agreement is not enough.** A model can report the stopped car ahead and
  still propose `KEEP_LANE` at cruise: check the fields against each other too.
- **Confidence has carried no measurable signal** in any model measured so far.
  Test that before you rely on it. It earns no marks of its own, and closing it
  as an empty investigation, with evidence, is a complete answer.
- **The two models fail differently, and both reach your arbiter.** SmolVLM
  invents hazards the scene does not hold, and proposes actions the lanes do not
  allow. Qwen finds most vehicles but invents hazards too, more or fewer by
  prompt, and slows down where a clear lane beside it calls for a lane change.
  Hazard agreement catches the inventions, but neither SmolVLM's actions nor
  Qwen's caution, so an arbiter needs both checks on
  `request.scene`: the hazards against the measured objects, and the action
  against the lanes, which ones exist beside the car and which the measured
  objects leave clear.

### Two failures in opposite directions

Two recorded wrong answers decode cleanly, replay on any machine, and each
drives a closed loop toward a stopped car. They are your starting exercises.
Event logs are appended to, so give every run a new one:

```bash
uv run metadrive-starter run --config configs/demo-vla-semantic-stopped-vehicle.yaml --submission submission --event-log tmp/semantic-stopped-vehicle-01.jsonl
uv run metadrive-starter run --config configs/demo-vla-semantic-partial-occlusion.yaml --submission submission --event-log tmp/semantic-partial-occlusion-01.jsonl
uv run python -m metadrive_starter.hazard_agreement tmp/semantic-stopped-vehicle-01.jsonl tmp/semantic-partial-occlusion-01.jsonl
```

- **Stopped vehicle:** the answer reports the stopped car ahead, high risk, and
  proposes `KEEP_LANE` at cruise. Hazard agreement scores it as agreeing; only
  checking the action and speed against the hazards it reported catches it.
- **Partial occlusion:** the answer reports the truck, low risk, and misses the
  stopped car behind it. Its fields agree with each other, so checking them
  against each other finds nothing; hazard agreement scores it as disagreeing.

In these two scenes, compare `vla_metrics.authority_steps`,
`arbitrations_endorsed`, `arbitrations_restricted`, and `arbitrations_declined`
before and after each change: they show your arbiter's effect, and the driving
figures do not.

<details>
<summary>What the shipped arbiter does on them, and the confidence AUROC</summary>

With the shipped arbiter, the model's command holds authority on 199 of 200
steps in both, and only the safety floor stops the car, 3.0 m short
(`minimum_safety_gap_m`). Declining hands the same approach to the local
fallback, which the floor stops 3.0 m short too.

The last command also gives each log's confidence AUROC, the chance that an
agreeing assessment is more confident than a disagreeing one, where 0.5 is
chance: n/a in each, since each log repeats one answer. Pooled, the two would
give 1.00 only because their answers differ in confidence as well as in
agreement, so never pool runs from different sources.

</details>

## Probing your observation

The probe asks the model once about each of eight fixed scenes and scores the
answers, open-loop: nothing is applied to a vehicle. With `--submission`, your
`observation.py` and `agent.yaml` build each scene's observation, saved exactly
as sent. Use a new output directory every time; the probe never overwrites one:

```bash
uv run metadrive-starter probe --config configs/vla-probe-smolvlm-llama.yaml --submission submission --infer --seed 1 --output-dir tmp/probe-smolvlm-01
```

`summary.json` scores each layer as scenes passed out of scenes scored, in
`scores`:

| Layer | Passes when | Marked in |
| --- | --- | --- |
| a scene's `status` | `success` when the answer decoded; `failure` with a decoder error in `metadata.json` when it did not, and also when no answer came back: a timeout or transport error, which marking does not count, or a request the server refused, which it counts as an answer that does not decode | [syntax](#syntax-10-points) |
| `perception` | the answer reports every hazard the scene requires | close to the recall in [scene interpretation](#scene-interpretation-15-points) |
| `tactical_action` | the proposed action is one the scene accepts | [tactical decision](#tactical-decision-30-points) |
| `raw_speed` | the proposed speed is in the scene's range; diagnostic, since the runtime maps each action to its own speed | [tactical decision](#tactical-decision-30-points) |

These scores are close to the marks, not the same. Scene interpretation is
marked per hazard, on scenes of its own: `scores.perception` passes a scene
whose answer reports every hazard it requires, which is close to recall, but it
never checks for invented hazards, which cost precision, and it requires road
features, which marking never counts.
[`probe-analysis.ipynb`](probe-analysis.ipynb) lists each hazard an answer
invents as "not in the scene", and pools a run's answers into recall, precision,
and F1 as marking does. The tactical-decision mark also needs the speed
to fit the scene, before and after your arbiter, which the probe never runs.

What each scene accepts is in [`configs/vla-probe-scenarios.yaml`](../../configs/vla-probe-scenarios.yaml). Against a
stopped vehicle in your lane, the right answer is a change into a clear
adjacent lane, and waiting, with `STOP` or `FOLLOW`, only when every lane is
blocked; `SLOW_DOWN` still closes on it, so it never passes there. The shipped
prompt policy already says so. Assignment 2's configurations enable lane
changes, which execute once the safety floor finds the target lane clear.

[Evaluation](../../docs/evaluation.md#open-loop-vla-probe) lists the files a probe run
writes for each scene. The recorded scenes in
`fixtures/vla/probes/catalog-20261002-v5/` replay with `--replay-from` without
MetaDrive: alone they send DriveBench's original observation, and with
`--submission` yours, built from the recorded frame and scene.

**The probe places only cars and trucks**, so none of its scenes holds a
pedestrian, a cone, or a barrier, and
[scene interpretation](#scene-interpretation-15-points) marks those too. Eight
recorded scenes in `fixtures/vla/probes/interpretation-public-20261030/` hold them, at
several distances and lanes, on maps the hidden scenes do not use, each with a
`truth.json` that lists what was placed and whether the camera shows it. Ask a
model about them through your observation, with the profile scene interpretation
uses:

```bash
uv run metadrive-starter probe --config configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml --submission submission --infer --seed 1 --replay-from fixtures/vla/probes/interpretation-public-20261030 --output-dir tmp/probe-interpretation-01
```

List the output directory in `PROBE_RUNS` in
[`probe-analysis.ipynb`](probe-analysis.ipynb), which scores the answers against
that truth; the [observation inspector](#the-observation-inspector) shows what
your observation sends on them.

[`probe-analysis.ipynb`](probe-analysis.ipynb) reads probe output by those
layers, by hazard agreement, and by the measures marking pools, reads closed-loop
event logs, and puts each answer through your arbiter; with no runs of your own,
it replays the two recorded failures on every scene.

**The probe's traffic lights are enlarged**, to 1.75 times their size, so that
their state is legible at 512x288. A traffic-light result on the probe does not
generalise to real signals. Traffic lights are not marked in this assignment:
no hidden scene holds one, so the public red-light scene is for practice only.

## Measuring and citing

Name, for every figure you report, the field it was read from and the command
that wrote it, for example "`scores.perception` in
`tmp/probe-qwen-03/summary.json`, from `probe --config
configs/vla-probe-qwen3-vl-2b-llama-structured.yaml --submission tmp/attempt-2
--infer --seed 3 --output-dir tmp/probe-qwen-03`". The probe's `metadata.json`
and `run`'s JSON summary give every figure at full precision; `gates` gives
none to cite.

A live model can answer differently every time. Report a live figure as the
median and range over your runs, with how many there were, and keep every
output directory you cite. Pool runs only when they sent the same observation
to the same model. Figures from the recorded answers and scenes reproduce
exactly.

<details>
<summary>What each tool covers</summary>

| Tool | Covers | Does not cover |
| --- | --- | --- |
| observation inspector | what your `observation.py` sends on the eight probe scenes and the eight recorded scenes with pedestrians and obstacles | any model's answer |
| `probe --submission ... --infer` | one answer per scene from one model, scored per layer | your arbiter; a closed loop |
| probe-analysis notebook | probe output by criterion, hazard agreement against the measured scene, recall, precision, and F1, SmolVLM's accepted answers, your arbiter on each answer, reruns, event logs | anything you did not give it |
| `run ... --event-log` | a closed loop: the summary's `vla_metrics`, and one event per decision and control tick | scores against a scene's expected answer |
| `python -m metadrive_starter.hazard_agreement` | the hazards and confidence of every assessment in event logs | probe output |
| `gates`, Assignment 2's manifest | the structural floor, with a recorded answer | model quality, and your Assignment 2 marks |

</details>

<details>
<summary>Measuring an attempt you do not keep</summary>

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

</details>

## Check your work

Every submission must pass all of Assignment 2's structural gates. They catch
what breaks a run, never how well it drives, and they need no model server:
they drive your controller, observation, and arbiter together with a recorded
model answer.

```bash
uv run metadrive-starter gates --manifest configs/gates-assignment-2.yaml --config configs/demo-vla-fixture.yaml
```

| Gate | Passes when |
| --- | --- |
| submission loads | every file imports and defines its entry point with the right methods, and `agent.yaml` sets only permitted keys within their bounds |
| bounded control outputs | every control action is two finite numbers within `[-1, 1]` |
| no run stops on an error | no route stopped on an error; a route that crashes, leaves the road, or does not arrive is reported and fails no gate, since Assignment 2 does not mark driving |
| valid observations emitted | every observation built is an `Observation` with an `RGBFrame`, non-empty text, and the captured timestamp, and so is each request built again for Qwen and for SmolVLM, with `request.model_id` naming each; its detail counts any without the output contract, which never fails it |
| control loop never blocked | no `build` took longer than the request timeout, 1 s in this configuration, and no arbiter call longer than one control step; a call that overruns is abandoned |
| monotone restriction preserved | no `arbitrate` granted more authority than the model proposed or answered with anything but an assessment, no `review` answered anything but True or False, and neither raised |
| faults never break the run | no fault scenario stopped on an error; a fault scenario need not arrive, and a crash or off-road exit is reported and fails no gate |
| nothing outside `submission/` was modified | every file shipped outside `submission/`, notebooks aside, is unchanged; this gate runs only in a student release |

Grading runs the same command against hidden routes and withheld fault
scenarios. Passing the public gates is a close reference for those, not a
guarantee: code that raises on a road or a fault timing it never met fails
there.
`--scenario camera-dropout` drives one scenario while iterating and marks the
report partial; before handing in, run them all.

## What you hand in

A zip of your `submission/` folder, uploaded to Moodle by the deadline, as
[Hand in](../../docs/getting-started.md#hand-in) describes. For Assignment 2 it
holds:

| File | What it holds |
| --- | --- |
| `observation.py` | your observation: what the model receives for each request |
| `arbitration.py` | your arbiter: what each assessment keeps of the model's authority |
| `agent.yaml` | your tuning values, Assignment 1's and the [ones above](#the-values-in-agentyaml) |
| `controller.py` | your Assignment 1 controller; the gates drive with it, and nothing in Assignment 2 marks it |

Your `submission/` carries forward: keep your Assignment 1 `controller.py` and
`agent.yaml` beside the `observation.py` and `arbitration.py` stubs that
shipped with them. To check the zip, run the gates of
[Check your work](#check-your-work) on its unpacked copy.

## How you are marked

Assignment 2 is marked out of 100, in two parts. Hidden runs of the open-loop
probe, still scenes, one frame and one answer at a time, compute points, and
those points carry most of the mark. Staff then review every submission,
reading its code against those runs, and that review is the rest of the mark:
points the code does not account for are not earned by the runs alone, so a
number tuned against the public scenes is not a mark. Write code a reader can
follow: document each rule and each value where it is written, and say what it
is for. The vehicle is never driven for a
mark; a live drive is for your own exploration and the oral defence. There is
no written report: your team explains and analyses its work at the oral
defence.
Teams never produce a graded number: public scenes, recorded answers, and
public faults are for your own measurements. This marking does not change once
Assignment 2 is released: if a correction is ever needed, it is announced and
applies to every team alike.

The shipped files pass every gate. The shipped observation is DriveBench's
original less one line, which a `TODO` replaces, and the shipped arbiter
endorses everything, so it scores little on tactical decision, only from answers
that arrive below the minimum confidence.

### The floor: hidden structural gates

Grading runs [the structural gates](#check-your-work) on your
`submission/` against hidden routes and withheld fault scenarios, with the same
command and code you run on the public ones.

- If every gate passes, or reports "not run", the assignment is marked from the
  hidden runs below.
- If any gate fails, the assignment earns nothing.

The gates you run are the gates that grade you, on routes and faults you have
not seen, so passing them in public is a close reference, not a guarantee. How
your Assignment 1 controller drives fails none of them: a route it crashes on
or leaves costs nothing here. You may replace your upload on Moodle until the
deadline, and only the last upload is graded; once the deadline closes there
are no resubmissions. The deadlines leave time to run every check before
handing in. A failure caused on the grading side is graded again by staff.

Faults are part of the floor, not of any mark. The public gate run schedules
one scenario of each of the six published kinds; grading uses other timings,
maps, and combinations, two at once. Your arbiter has only to survive them, so
this is what each hands it:

| Fault | What `arbitrate` sees |
| --- | --- |
| `camera_dropout` | no new request, so no assessment; the command that holds authority grows old |
| `lidar_dropout` | with LiDAR as the safety source, a scene with `valid` false and no objects |
| `provider_timeout` | no assessment |
| `provider_error` | no assessment |
| `provider_latency` | an answer that arrives late; `now_s` minus `issued_at_s` is its age |
| `malformed_output` | no assessment |

The three provider failures reach only `review`, as `request.failure`, which
Assignment 2 leaves returning True, so each leaves the last command you let
through to drive until its action horizon ends. Retrying belongs to
Assignment 3.

### The hidden probes

Every point comes from runs on instructor hardware, never from a number a
team produces:

- **The models** are the pinned Qwen3-VL-2B, `Q4_K_M`, and SmolVLM-256M,
  `Q8_0`, under the pinned `llama-server`, as in
  [`configs/vla-probe-qwen3-vl-2b-llama.yaml`](../../configs/vla-probe-qwen3-vl-2b-llama.yaml),
  [`configs/vla-probe-qwen3-vl-2b-llama-structured.yaml`](../../configs/vla-probe-qwen3-vl-2b-llama-structured.yaml),
  [`configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml`](../../configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml), and
  [`configs/vla-probe-smolvlm-llama-structured.yaml`](../../configs/vla-probe-smolvlm-llama-structured.yaml). Each samples at
  temperature 0.7, as those profiles do, with the seed fixed to a different
  value on each rerun and the server's prompt cache off (`--no-cache-prompt`),
  so that each answer depends only on its request and seed. Reruns therefore
  differ the way your own runs do, and every grade can be reproduced on the
  machine that made it.
- **Hidden scenes** are of two kinds, and your `observation.py` builds each
  observation, after your `agent.yaml`, as `probe --submission` does, with the
  objects and traffic lights withheld from the request as in every Assignment 2
  run, and for Qwen on the interpretation scenes the lane clearance too.
  - **Hidden probe scenes**, for syntax and tactical decision, are of the kinds
    the public catalog holds, but for the traffic light: clear roads, curves,
    intersections, stopped, adjacent, and partly hidden vehicles, and lanes
    blocked ahead. Each declares the actions it accepts and a target-speed
    range, as the public scenes in
    [`configs/vla-probe-scenarios.yaml`](../../configs/vla-probe-scenarios.yaml) do.
  - **Hidden interpretation scenes**, about forty, for scene interpretation:
    empty roads, straight, bending, and at junctions; a car, truck, pedestrian,
    traffic cone, or barrier at several distances in your lane and in the lanes
    beside it; two and three objects at once; and objects off your lane,
    including one beside the car, out of the camera's view. Each records which
    of its objects the camera shows.
- **Withheld assessments** are recorded assessments, sound and unsound, of the
  kinds the two semantic-failure fixtures show, and confident answers SmolVLM
  gave on the grading hardware. They are given to your arbiter on hidden probe
  scenes beside the models' own.
- **Public figures are a close reference, not a forecast.** The hidden scenes
  use other maps, seeds, and distances, so a submission tuned to the public
  scenes can score lower on the hidden ones; your public figures show what to
  expect, never what your marks will be.
- **Five reruns.** Each measurement below is run five times. Your marks use the
  median; the report gives each run's figure, the median, and the variance
  across the five. A submission that is stable across reruns does better than
  one that sampled well once.

### The mark

Four measures earn 60 points between them, and the computed points are their
total scaled to 100, which the staff review above then completes: syntax counts
for a sixth of it, scene interpretation a quarter, SmolVLM's accepted answers a
twelfth, and tactical decision half.

#### Syntax: 10 points

With structured output off, so that the model is held to the schema only by
your observation: the share of Qwen's answers to the hidden probe scenes that
decode. A request that times out or fails in transport returns no answer and is
not counted. A request the server refuses, as it can refuse an observation too
long for it, counts as an answer that does not decode. A run with no answer at
all counts 0.

Points: 10 × the median share.

#### Scene interpretation: 15 points

With structured output on, so that a format failure is not paid for twice:
how well Qwen's answers to the hidden interpretation scenes report the objects
the camera shows, hazard by hazard. Each vehicle, pedestrian, obstacle, or
traffic control an answer reports is a claim, paired with the scene as
DriveBench measured it, as `metadrive_starter.hazard_agreement.score_hazards`
pairs them: each claim with one measured object of its type, at most one
45-degree sector away, exact sectors first. A claim left over is invented, so
naming one vehicle twice invents one. A road feature, or a hazard of type
`other`, cannot be measured, so neither is ever counted.

- **Recall** is the share of the objects to be found that a claim paired with.
  An answer must find every object the camera shows within the measured scene's
  50 m radius. An object out of view, beside or behind the car, need not be
  found, and a claim of it is not invented.
- **Precision** is the share of claims that paired. An invented hazard costs
  precision, not the scene.
- Each rerun counts every answer at once, and its figure is their F1:
  2 × precision × recall ÷ (precision + recall).

An answer that does not decode claims nothing and finds nothing. Requests that
fail or are refused count as for syntax.

Guessing earns an F1 too, so grading also finds, without a model, the best
answer that is the same on every scene: of every list of up to three vehicles,
pedestrians, obstacles, or traffic controls, each at `front`, `front_left`, or
`front_right`, the one with the highest F1 on the hidden interpretation scenes.
That F1 is the floor. The grade report names that answer and shows its figures
beside yours, with those of three plain answers: no hazard; a vehicle at
`front`; and a vehicle at each of `front`, `front_left`, and `front_right`. You
can score your answers, and any constant answer, on
[the recorded public scenes](#probing-your-observation) with `score_hazards`, as
[`probe-analysis.ipynb`](probe-analysis.ipynb) does; there are few of them, so
they show what your prompt misses, not the floor or your F1. Your points pay the
margin your answers earn above the floor, in full at an F1 of 0.80, which a
well-built prompt reaches on scenes like these.

Points: 15 × (median F1 − floor) ÷ (0.80 − floor), never below 0 and never
above 15. An F1 at the floor or below earns nothing; one of 0.80 or more earns
all 15.

The report also gives each rerun's recall and precision beside its F1, so you
can see whether your answers miss objects or invent them. Neither earns points
of its own.

#### SmolVLM answers the runtime accepts: 5 points

With structured output on, on every hidden scene of both kinds: the share of
SmolVLM's answers that the runtime accepts, to the observation your
`observation.py` builds when `request.model_id` names SmolVLM. An answer is
accepted when it decodes and its `confidence` is at or above the confidence
floor, `command_validation.minimum_confidence` in the SmolVLM profile's
configuration, 0.5. `uv run metadrive-starter config --config
configs/vla-probe-smolvlm-llama-structured.yaml` prints it. The runtime discards
a command below the floor, so an answer that decodes but would never be acted on
earns nothing. The floor marked is the profile's own: raising yours in
`agent.yaml` tightens your arbiter, never this measure.

The grammar guarantees an answer's shape, not its values: every answer takes
the assessment's form, but the decoder still refuses one whose numbers fall
outside the schema's ranges, and nothing in the grammar keeps a confidence at
or above the floor. A prompt that suits Qwen can cost SmolVLM most of its
answers. SmolVLM answers what it is shown, so look at what it actually answers
to your prompt, and at why each answer was refused or discarded: a probe's
`metadata.json` gives each refused answer's reason in `outcome.error_message`
and each decoded answer in `outcome.parsed_assessment`,
[`probe-analysis.ipynb`](probe-analysis.ipynb) counts a run's accepted answers
against the floor, and a closed-loop run's summary counts the floor's reasons in
`vla_metrics.validation_reasons`. Then change the prompt until its answers are
ones the runtime keeps. Requests that fail or are refused count as for syntax.
SmolVLM's answers to the hidden probe scenes also go to your arbiter, beside
Qwen's; how sound they are earns no mark of its own.

Points: 5 × the median share.

#### Tactical decision: 30 points

With structured output on. The assessments form one pool: Qwen's and
SmolVLM's answers to the hidden probe scenes, and the withheld ones. Nothing is
marked per model, so an unsound SmolVLM answer counts as an unsound Qwen answer
does. Each assessment goes through your `arbitrate` in a fresh episode, as in a
closed loop: `begin_episode()`, then that one assessment, with its still scene
stamped at the time it arrives, as the live loop hands your arbiter the scene
measured now. It arrives late, as in a closed loop. A model's own
answer arrives as late as the grading server takes to write it with its GPU to
itself, counted from the tokens the server reports in `usage` rather than by the
clock, so the server's load never moves your mark: 0.34 s, plus 0.18 ms for each
token of the request and 5 ms for each token of the answer, for either model. A
longer answer arrives later. A withheld one arrives after the delay recorded
with it. Because every assessment starts a new episode, history kept across
assessments neither helps nor hurts here.

Each assessment is then classified:

- an assessment is **sound** when its action is one the scene accepts and its
  target speed is within the scene's range, and **unsound** otherwise;
- your arbiter **catches** an unsound assessment when it declines it, restricts
  it into a sound one, or returns it below your `minimum_confidence`, which the
  floor rejects. That includes one that arrived below it and was endorsed;
- it **keeps** a sound assessment when what it returns is still sound and at or
  above your `minimum_confidence`. Declining keeps one only on a scene that
  accepts `REQUEST_FALLBACK`.

An `arbitrate` answer that cannot stand, because the call raised or granted
more authority than the model proposed, counts as declined, and fails the
"monotone restriction preserved" gate as well. The withheld assessments include
sound and unsound ones, so both rates below are always defined.

The tactical score is the catch rate plus the keep rate, minus one, and never
below zero:

```text
score = max(0, caught / unsound + kept / sound - 1)
```

Endorsing everything, as the shipped arbiter does, keeps every sound assessment
that arrives at or above your `minimum_confidence` and catches only the unsound
ones that arrive below it, so with models that answer confidently it scores 0.
Declining everything scores 0 too. Only an arbiter that tells the two apart
earns marks.

Points: 30 × the median score.

`review` is not part of Assignment 2, and is never called here.

## Rules

- Change only files in `submission/`. The simulator, the model providers, the
  safety floor, and the gates are course code that your files call but never
  edit.
- Import only what the course environment already provides: the Python standard
  library, `metadrive_starter`, and its installed dependencies. Grading
  installs nothing of yours.
- Keep every tuned value that has a [permitted key](#the-values-in-agentyaml)
  in `agent.yaml`, and any other, such as a cutoff your arbiter's rules use, as
  a named constant at the top of the file that uses it, where a reader finds it.
  Your files read nothing but what DriveBench passes them: no other files, no
  environment variables, no network. Your observation never calls a model, and
  your arbiter never asks one.
- Measure on public scenes, recorded answers, or maps and seeds you choose
  yourself. Hidden scenes are run only by the instructors.

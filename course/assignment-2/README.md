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

**What Assignment 2 marks.** The marks come from still scenes: one frame, one
observation, one assessment at a time, each in a fresh episode, as
[How you are marked](#how-you-are-marked) says. Your observation and arbiter
also sit in the live driving system, and the gates drive routes and fault
scenarios with them, but only to check that nothing breaks: how well the car
drives earns nothing, history your arbiter keeps across assessments earns
nothing, and `review` stays as shipped. Drive live to explore your work and to
show it at the oral defence.

## Steps

1. [Set up](#set-up), and check that the shipped files pass Assignment 2's
   gates.
2. [Serve SmolVLM](#serving-the-models) and [probe](#probing-your-observation)
   the shipped observation with it.
3. Work through [the task](#the-task) with the two notebooks,
   [measuring](#measuring-and-citing) every change on SmolVLM and confirming
   it on Qwen, which earns most of the marks.
4. [Check your work](#check-your-work) with the gates.
5. [Hand in](#what-you-hand-in) a zip of `submission/` on Moodle.

## Set up

```bash
uv sync --group dev --extra notebooks
uv run metadrive-starter gates --manifest configs/gates-assignment-2.yaml --config configs/demo-vla-fixture.yaml
uv run --extra notebooks jupyter lab course/assignment-2/observation-inspector.ipynb
```

Every gate should report `PASSED`. The gates use a recorded model answer, so
they need no model server; they take a few minutes, longer the first time in a
new clone or folder while MetaDrive downloads its assets.

## The task

Every gap in the shipped [`submission/observation.py`](../../submission/observation.py) and
[`submission/arbitration.py`](../../submission/arbitration.py) is marked by a named `TODO` line, and each
file's docstring states the contract it must keep. The two notebooks take the
`TODO` lines as their tasks, and end each task with the structural gates' own
checks, printed as `PASS` or `NOT YET`:

1. **[Your observation](#your-observation)**: what the model receives, and
   where each fact goes: in the text, drawn into the image, or left for the
   model to read from the pixels. The
   [observation inspector](observation-inspector.ipynb) takes its `TODO` lines
   as tasks 1 to 3: the instructions, which hold two, the facts the text gives,
   and the image.
2. **[Your arbiter](#your-arbiter)**: how far to believe each assessment, the
   model's answer. Catch the unsound assessments and keep the sound ones, while
   only ever taking authority, a command's permission to influence the
   vehicle, away. [Probe analysis](probe-analysis.ipynb) takes its `TODO` lines
   as one task.

The model's answers to your observation earn the syntax, SmolVLM, and
scene-interpretation marks, and your arbiter earns the tactical-decision mark.
Grading asks Qwen3-VL-2B for all of them but SmolVLM's 5 points, so probe your
observation on Qwen too, with structured output off and on. Each request names
the model it goes to, so your observation can prompt each model its own way.

Throughout, **keep a record** of what you try and measure, with the command
behind each figure: there is no written report, but at the oral defence your
team explains and analyses its work. Analysis worth doing as you go, with
measurements: why SmolVLM, which follows a one-word instruction, does not
follow your observation without the grammar; on which scenes your observation
makes SmolVLM's commands sound, and why not on others, though no mark depends
on how sound they are; and whether a model's confidence carries signal your
arbiter can use. Some of these have no fix, and concluding so with evidence is
a sound result.

## The model ladder

You work with two models, from the pinned files
[Model providers](../../docs/model-providers.md#the-pinned-llama-server-and-model-files)
lists: SmolVLM-256M, `Q8_0`, the instrument you develop against; and
Qwen3-VL-2B, `Q4_K_M`, the stronger profile. Syntax and scene interpretation
are measured on Qwen's answers, the five SmolVLM points on SmolVLM's, and
tactical decision on one pool of both models' answers and withheld ones, as
[How you are marked](#how-you-are-marked) states.
A server answers either unconstrained, or under JSON Schema output, also called
structured output, where a grammar lets the model write only text that fits the
assessment's schema. DriveBench's decoder then turns the text into an
assessment; an answer it refuses does not decode.

**SmolVLM is a teaching instrument, not a smaller Qwen.** Without a grammar it
answered every probe request with a prose caption of the image, so no answer
decoded. With the grammar every answer takes the assessment's form, whatever it
understood: a cheap, local source of well-formed, wrong assessments, exactly
what arbitration exists to catch. What is marked is
[SmolVLM with the grammar on](#smolvlm-answers-the-runtime-accepts-5-points).
A way in: give SmolVLM a prompt of its own, probe it with
[`configs/vla-probe-smolvlm-llama-structured.yaml`](../../configs/vla-probe-smolvlm-llama-structured.yaml),
read why each answer was refused or discarded (`outcome.error_message` and
`outcome.parsed_assessment` in each scene's `metadata.json`;
[`probe-analysis.ipynb`](probe-analysis.ipynb) counts the accepted ones), and
check that Qwen's answers did not change.

### Serving the models

Serve one model at a time on port 8000 with the pinned `llama-server`, as
[Model providers](../../docs/model-providers.md#serving-a-model-for-assignment-2)
describes for your platform and hardware, then pass its configuration with
`--config`:

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
does. Under one seed the same request gets the same answer, provided the server
reads every prompt in full: serve with `--no-cache-prompt`, as grading does, or
a first answer can differ from later ones that reuse its cached prompt. Rerun
with `--seed 1`, `--seed 2`, and so on, and report every live figure as the
median and range over those runs. On
Apple silicon, do all marked work on the pinned build on Metal, never MLX, whose
figures do not predict your marks. Qwen is about ten times slower than SmolVLM,
so run it on your team's strongest laptop, together. Your laptop changes how you measure, never your mark: grading
ages each answer by its token counts, as
[Tactical decision](#tactical-decision-30-points) states, and a live run's
realtime factor is chosen and reported as
[Running and configuring](../../docs/running-and-configuring.md#choosing-a-realtime-factor)
says.

## Your observation

[`submission/observation.py`](../../submission/observation.py) defines `ObservationBuilder(settings)`. For every model
request, DriveBench calls `build(request)` and sends the model exactly the
`Observation` it returns, one `RGBFrame` and the prompt text. The request
carries the frame, the ego and cruise speeds, the parts of the measured scene
the model may have, and `model_id`, the model it goes to. Which facts to
include, and at what fidelity, is yours.

**The image is the model's only source of objects.** Assignment 2's requests
withhold the measured objects and traffic lights, as `request.withheld` names:
`request.scene` holds which lanes beside yours exist, whether each is observed
clear, and the distances to the road's edges. The hidden interpretation scenes
withhold lane clearance too, since it says on which side an object stands:
`request.withheld` names `lane_clearance` and both lanes' clearance is `None`,
so build an observation that works without it. Your arbiter still receives the
whole measured scene.

- Keep the frame's `timestamp_s`: crop, scale, or draw on the image, but never
  re-date it.
- `build` must return within 1 s, the gates' request timeout and grading's. The
  probe configurations allow 60 s, so a slow `build` passes every probe and
  fails the ["control loop never blocked"](#check-your-work) gate.
- **The output contract is a recommended default.** `request.contract` holds
  four lines that ask for exactly what the decoder accepts. Keep each verbatim
  on a line of its own, or rephrase or replace them: a run only counts an
  observation without them, in `vla_metrics.observations_without_output_contract`.
  But a model the grammar does not hold rarely answers in a form the decoder
  accepts, and that is what your syntax mark measures.

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

Any other key, or a value outside its bounds, is rejected by name; bounds count
from the configuration a run passes with `--config`, as
[Running and configuring](../../docs/running-and-configuring.md#configuration)
shows. The observation toggles act only through an `observation.py` that reads
them, as the shipped one does. `command_validation` holds the safety floor's
thresholds, which your arbiter also receives as its `settings`: you may make
the floor stricter, never looser.

## Your arbiter

[`submission/arbitration.py`](../../submission/arbitration.py) defines `Arbiter(settings)`, with the
`command_validation` thresholds as its `settings`;
[VLA pipeline](../../docs/vla-pipeline.md#arbitration) gives the whole contract.
`arbitrate(request)` sees each assessment as it arrives, on the control loop,
so it must return within one control step, 0.1 s at the default timing;
`begin_episode()` runs before each episode's first assessment, to clear any
history. Assignment 2 asks for `arbitrate` only: leave `review` returning True.

**You may only take authority away**, the monotone restriction. `arbitrate` may
lower `proposed_target_speed_mps`, lower `confidence` (below
`settings.minimum_confidence` the floor falls back), or decline by replacing
`proposed_action` with `REQUEST_FALLBACK`, which hands control to the local
fallback, driving at the cruise speed with the safety floor. Any other answer,
or a call that raises, is overruled, and the "monotone restriction preserved"
gate fails; an exception from `begin_episode()` stops the run.

The shipped arbiter endorses everything. The evidence so far:

- **Hazard agreement carries signal.**
  `metadrive_starter.hazard_agreement.score_hazards` scores the hazards an
  assessment reports against `request.scene`, quickly enough for every call.
- **Agreement is not enough.** A model can report the stopped car ahead and
  still propose `KEEP_LANE` at cruise: check the fields against each other too.
- **Confidence has carried no measurable signal** in any model measured so far.
  Test that before you rely on it; closing it as an empty investigation, with
  evidence, is a complete answer.
- **The two models fail differently, and both reach your arbiter.** SmolVLM
  invents hazards and proposes actions the lanes do not allow; Qwen finds most
  vehicles but invents hazards too, and slows down where a clear lane beside it
  calls for a lane change. Check the hazards against the measured objects, and
  the action against the lanes that exist and that those objects leave clear.

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
  stopped car behind it. Its fields agree with each other; hazard agreement
  scores it as disagreeing.

Compare `vla_metrics.authority_steps`, `arbitrations_endorsed`,
`arbitrations_restricted`, and `arbitrations_declined` before and after each
change: they show your arbiter's effect, and the driving figures do not.
[Evaluation](../../docs/evaluation.md#the-two-recorded-failures) gives what the
shipped arbiter does on these two.

## Probing your observation

The probe asks the model once about each of eight fixed scenes and scores the
answers, open-loop: nothing is applied to a vehicle. With `--submission`, your
`observation.py` and `agent.yaml` build each observation, saved exactly as
sent. Use a new output directory every time; the probe never overwrites one:

```bash
uv run metadrive-starter probe --config configs/vla-probe-smolvlm-llama.yaml --submission submission --infer --seed 1 --output-dir tmp/probe-smolvlm-01
```

Its `summary.json` scores are close to the marks, not the same, as
[Evaluation](../../docs/evaluation.md#probe-scores-and-assignment-2s-marks) says;
[`probe-analysis.ipynb`](probe-analysis.ipynb) pools a run's answers into
recall, precision, and F1 as marking does, and puts each through your arbiter.
What each scene accepts is in
[`configs/vla-probe-scenarios.yaml`](../../configs/vla-probe-scenarios.yaml):
against a stopped vehicle in your lane, a change into a clear adjacent lane,
and `STOP` or `FOLLOW` only when every lane is blocked; `SLOW_DOWN` still
closes on it, so it never passes there. Traffic lights are not marked: no
hidden scene holds one.

The probe places only cars and trucks, and
[scene interpretation](#scene-interpretation-15-points) marks pedestrians,
cones, and barriers too. Eight recorded scenes in
`fixtures/vla/probes/interpretation-public-20261030/` hold them, on maps the
hidden scenes do not use, each with a `truth.json` of what was placed and
whether the camera shows it. They replay without MetaDrive, through your
observation:

```bash
uv run metadrive-starter probe --config configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml --submission submission --infer --seed 1 --replay-from fixtures/vla/probes/interpretation-public-20261030 --output-dir tmp/probe-interpretation-01
```

List the output directory in `PROBE_RUNS` in
[`probe-analysis.ipynb`](probe-analysis.ipynb) to score it against that truth.
The eight probe scenes replay the same way from
`fixtures/vla/probes/catalog-20261002-v5/`.

## Measuring and citing

Name, for every figure you report, the field it was read from and the command
that wrote it, for example "`scores.perception` in
`tmp/probe-qwen-03/summary.json`, from `probe --config
configs/vla-probe-qwen3-vl-2b-llama-structured.yaml --submission tmp/attempt-2
--infer --seed 3 --output-dir tmp/probe-qwen-03`". Report a live figure as the
median and range over your runs, with how many there were, and pool runs only
when they sent the same observation to the same model. Measure each attempt in
a copy, `cp -R submission tmp/attempt-1`, never by editing and reverting
`submission/`. [Evaluation](../../docs/evaluation.md#measuring-assignment-2-work)
lists what each tool covers.

## Check your work

Every submission must pass all of Assignment 2's structural gates. They catch
what breaks a run, never how well it drives, and they drive your controller,
observation, and arbiter with a recorded answer:

```bash
uv run metadrive-starter gates --manifest configs/gates-assignment-2.yaml --config configs/demo-vla-fixture.yaml
```

| Gate | Fails when |
| --- | --- |
| submission loads | a file fails to import or lacks its entry point or methods, or `agent.yaml` sets a key or value it may not |
| bounded control outputs | a control action is not two finite numbers within `[-1, 1]` |
| no run stops on an error | a route stops on an error; a crash, an off-road exit, or an unfinished route fails no gate |
| valid observations emitted | an observation, built for Qwen or for SmolVLM, is not an `Observation` with an `RGBFrame`, non-empty text, and the captured timestamp |
| control loop never blocked | a `build` overruns the request timeout, 1 s here, or an arbiter call one control step; the call is abandoned |
| monotone restriction preserved | `arbitrate` grants more authority than the model proposed, answers with anything but an assessment, or raises, or `review` answers anything but True or False |
| faults never break the run | a fault scenario stops on an error; it need not arrive |
| nothing outside `submission/` was modified | a file shipped outside `submission/`, notebooks aside, changed; this gate runs only in a student release |

`--scenario camera-dropout` drives one scenario while iterating and marks the
report partial; before handing in, run them all.

## What you hand in

A zip of your `submission/` folder, uploaded to Moodle by the deadline, as
[Hand in](../../docs/getting-started.md#hand-in) describes: `observation.py`
and `arbitration.py`; `agent.yaml`, with Assignment 1's values and
[the ones above](#the-values-in-agentyaml); and your Assignment 1
`controller.py`, which the gates drive with and nothing in Assignment 2 marks.
To check the zip, run the gates on its unpacked copy.

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

Grading runs [the structural gates](#check-your-work) on your `submission/`
against hidden routes and withheld fault scenarios, with the same command and
code you run on the public ones. If every gate passes, or reports "not run",
the assignment is marked from the hidden runs below; if any gate fails, the
assignment earns nothing.

Passing the gates in public is a close reference, not a guarantee: code that
raises on a road or a fault timing it never met fails there. Faults are part of
the floor, not of any mark: grading uses other timings, maps, and combinations,
two at once, and your arbiter has only to survive them;
[Evaluation](../../docs/evaluation.md#what-an-arbiter-sees-under-each-fault) lists
what each hands it. You may replace your upload on Moodle until the deadline,
and only the last upload is graded; once the deadline closes there are no
resubmissions. A failure caused on the grading side is graded again by staff.

### The hidden probes

Every point comes from runs on instructor hardware, never from a number a
team produces:

- **The models** are the pinned Qwen3-VL-2B and SmolVLM-256M under the pinned
  `llama-server`, as in the probe profiles of
  [Serving the models](#serving-the-models). Each samples at temperature 0.7,
  with the seed fixed to a different value on each rerun and the server's
  prompt cache off (`--no-cache-prompt`), so that each answer depends only on
  its request and seed: reruns differ the way your own runs do, and every grade
  can be reproduced on the machine that made it.
- **Your observation** builds every request, after your `agent.yaml`, as
  `probe --submission` does, with the objects and traffic lights withheld, and
  for Qwen on the interpretation scenes the lane clearance too.
- **Hidden probe scenes**, for syntax and tactical decision, are of the public
  catalog's kinds but for the traffic light: clear roads, curves,
  intersections, stopped, adjacent, and partly hidden vehicles, and lanes
  blocked ahead. Each declares the actions it accepts and a target-speed range.
- **Hidden interpretation scenes**, about forty, for scene interpretation:
  empty roads, straight, bending, and at junctions; a car, truck, pedestrian,
  traffic cone, or barrier at several distances in your lane and beside it;
  two and three objects at once; and objects off your lane, including one
  beside the car, out of the camera's view. Each records which of its objects
  the camera shows.
- **Withheld assessments** are recorded assessments, sound and unsound, of the
  kinds the two semantic-failure fixtures show, and confident answers SmolVLM
  gave on the grading hardware, given to your arbiter on hidden probe scenes
  beside the models' own.
- **Public figures are a close reference, not a forecast.** The hidden scenes
  use other maps, seeds, and distances, so a submission tuned to the public
  scenes can score lower on the hidden ones.
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
the camera shows. Each vehicle, pedestrian, obstacle, or traffic control an
answer reports is a claim, paired with the measured scene as
`metadrive_starter.hazard_agreement.score_hazards` pairs them: with one
measured object of its type, at most one 45-degree sector away, exact sectors
first. A claim left over is invented, so naming one vehicle twice invents one.
A road feature, or a hazard of type `other`, cannot be measured, so neither is
ever counted.

- **Recall** is the share of the objects to be found that a claim paired with:
  every object the camera shows within the measured scene's 50 m radius. An
  object out of view, beside or behind the car, need not be found, and a claim
  of it is not invented.
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
That F1 is the floor. The grade report shows that answer and its figures beside
yours, with three plain answers: no hazard; a vehicle at `front`; and a vehicle
at each of `front`, `front_left`, and `front_right`. You can score any answer,
yours or a constant one, on
[the recorded public scenes](#probing-your-observation) with `score_hazards`, as
[`probe-analysis.ipynb`](probe-analysis.ipynb) does; there are few of them, so
they show what your prompt misses, not the floor or your F1.

Points: 15 × (median F1 − floor) ÷ (0.80 − floor), never below 0 and never
above 15: an F1 at the floor or below earns nothing, and one of 0.80 or more,
which a well-built prompt reaches on scenes like these, earns all 15. The
report also gives each rerun's recall and precision, which earn no points of
their own, so you can see whether your answers miss objects or invent them.

#### SmolVLM answers the runtime accepts: 5 points

With structured output on, on every hidden scene of both kinds: the share of
SmolVLM's answers that the runtime accepts, to the observation your
`observation.py` builds when `request.model_id` names SmolVLM. An answer is
accepted when it decodes and its `confidence` is at or above the confidence
floor, `command_validation.minimum_confidence` in the SmolVLM profile's
configuration, 0.5, which `uv run metadrive-starter config --config
configs/vla-probe-smolvlm-llama-structured.yaml` prints. The runtime discards a
command below the floor, so an answer that would never be acted on earns
nothing. The floor marked is the profile's own: raising yours in `agent.yaml`
tightens your arbiter, never this measure.

The grammar guarantees an answer's shape, not its values: the decoder still
refuses one whose numbers fall outside the schema's ranges, and nothing in the
grammar keeps a confidence at or above the floor, so a prompt that suits Qwen
can cost SmolVLM most of its answers; [The model ladder](#the-model-ladder)
says where to read why. Requests that fail or are refused count as for syntax.
SmolVLM's answers to the hidden probe scenes also go to your arbiter, beside
Qwen's; how sound they are earns no mark of its own.

Points: 5 × the median share.

#### Tactical decision: 30 points

With structured output on. The assessments form one pool: Qwen's and
SmolVLM's answers to the hidden probe scenes, and the withheld ones. Nothing is
marked per model, so an unsound SmolVLM answer counts as an unsound Qwen answer
does. Each assessment goes through your `arbitrate` in a fresh episode:
`begin_episode()`, then that one assessment, with its still scene stamped at
the time it arrives, as the live loop hands your arbiter the scene measured
now, so history kept across assessments neither helps nor hurts. It arrives
late, as in a closed loop. A model's own answer arrives as late as the grading
server, its GPU to itself, takes to write it, counted from the tokens the
server reports in `usage` rather than by the clock, so the server's load never
moves your mark: 0.34 s, plus 0.18 ms for each token of the request and 5 ms
for each token of the answer, for either model. A withheld one arrives after
the delay recorded with it.

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
sound and unsound ones, so both rates are always defined. The tactical score is
the catch rate plus the keep rate, minus one, and never below zero:

```text
score = max(0, caught / unsound + kept / sound - 1)
```

Endorsing everything, as the shipped arbiter does, keeps every sound assessment
that arrives at or above your `minimum_confidence` and catches only the unsound
ones that arrive below it, so with models that answer confidently it scores 0.
Declining everything scores 0 too. Only an arbiter that tells the two apart
earns marks.

Points: 30 × the median score. `review` is not part of Assignment 2, and is
never called here.

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

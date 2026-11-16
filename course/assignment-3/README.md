# Assignment 3: Request cadence and cloud operation

[← DriveBench README](../../README.md) · [Request timing](request-timing.ipynb)

> Assignment weight: **30%**

The course calendar gives the deadline, and late submissions are not accepted.
Run all commands from the repository root.

From now on the model is Gemini 3.5 Flash Lite, a cloud model reached through
Google's Vertex AI. Every request costs money, its answer takes about 1.4 s to
arrive, each drive may make only so many requests, and now and then the
provider fails or refuses. You decide when to ask: your request policy holds a
request back until asking is worth it, and keeps the car driving through an
outage. The [glossary](../../CONTEXT.md) defines the course's terms.

**What Assignment 3 marks.** The car drives. Grading drives hidden scenes, with
stopped vehicles and other obstacles, on Gemini, with your whole submission:
your Assignment 1 controller, your Assignment 2 observation and arbiter, and
your new request policy. Whether the car gets through, how many requests it
spends for nothing, how quickly it asks when the road changes, and how it
comes back after an outage earn the marks, as
[How you are marked](#how-you-are-marked) says. You may change your
Assignment 1 and 2 files; their marks never change.

## Steps

1. [Set up](#set-up), and check that the shipped files pass Assignment 3's
   gates.
2. Work through [the task](#the-task) with the
   [request timing](request-timing.ipynb) notebook, which needs no model.
3. [Practise on Vertex](#practising-on-vertex) with your team's project, on
   the demo and then [the practice scenes](#the-practice-scenes).
4. [Check your work](#check-your-work) with the gates.
5. [Hand in](#what-you-hand-in) a zip of `submission/` on Moodle.

## Set up

```bash
uv sync --group dev --extra notebooks
uv run metadrive-starter gates --manifest configs/gates-assignment-3.yaml --config configs/demo-vla-fixture.yaml
uv run --extra notebooks jupyter lab course/assignment-3/request-timing.ipynb
```

Every gate should report `PASSED`. The gates use recorded model answers, so
they need no model server and make no cloud request; they take a few minutes.

## The task

Every gap in the shipped [`submission/request_policy.py`](../../submission/request_policy.py) is marked by a named
`TODO` line, and the file's docstring states the contract it must keep. The
[request timing](request-timing.ipynb) notebook takes the `TODO` lines as four
tasks: ask when the scene changes; hold while the command in force still
holds; back off through faults; and keep a reserve of the budget. Each ends
with a check printed as `PASS` or `NOT YET`, together with the structural
gates' own checks of your policy.

`request_policy.py` defines `RequestPolicy(settings)`. DriveBench calls
`begin_episode()` before a drive's first tick, then `should_request(tick)` on
every control tick, before it captures a camera frame. Return True to ask the
model now, False to hold the request back. `settings` holds the drive's limits:
`request_budget`, `minimum_interval_s`, `request_timeout_s`, and
`action_horizon_s`. `tick` holds the scene measured now, the command that holds
authority and when the model was asked for it, why a request failed on this
tick, whether one is in flight, and the requests left in the budget;
[VLA pipeline](../../docs/vla-pipeline.md#request-policy) gives the whole
contract. The file's docstring says which module each type comes from:
`tick.scene` is the `LocalScene` your arbiter receives.

**You may only ask less**, the monotone restriction. True is a request, not a
guarantee: DriveBench still refuses while a request is in flight, sooner than
`minimum_interval_s` after the last one, and once the budget is spent. A
`should_request` that raises, or returns anything but True or False, holds that
tick's request back and fails the "monotone restriction preserved" gate. Each
call must return within one control step, 0.1 s.

The shipped policy asks every time DriveBench lets it: it spends many requests
on a road that has not changed, and asks on through an outage.

### Between two requests

- **An answer takes time.** Gemini's answers took about 1.4 s in the course's
  drives; `cloud-check`, under
  [Practising on Vertex](#practising-on-vertex), measures yours. The command
  derived from an answer holds authority for `settings.action_horizon_s`,
  counted from the request, unless the safety floor refuses it, your arbiter
  declines it, or your arbiter's `review` ends it. In the hidden drives a
  command holds for 4.0 s, and an answer more than 2.5 s old when it arrives
  is refused.
- **When no command holds,** the local fallback drives: at the cruise speed,
  with the safety floor and emergency braking, up to whatever stands in the
  lane, where the safety supervisor stops the car about 3 m short.
- **A lane change needs room.** The safety floor refuses a lane change while
  an object ahead is closer than 12 m. A car that stops closer than that behind
  a stopped object can never change lanes, and stays stuck. A lane change has
  to be asked for, answered, and started before the car is that close.
- **During a fault no code of yours can command the car.** No answer arrives,
  the command in force ends at its horizon, and the local fallback takes over.
  What your policy controls is how much budget it spends on failures, and how
  soon it finds the provider back.
- **Every request the provider receives costs budget**, whether it is
  answered, fails, times out, or is refused for quota or an outage. Once the
  budget is spent, the model is never asked again that drive.

`tick.failure` says why a request failed, on the tick its failure arrives:
`model_unavailable` during an outage, `quota` for a quota refusal, and
`timeout`, `invalid_response`, or `transport` otherwise.

### Your observation and arbiter

Your Assignment 2 files drive here too, and you may improve them. As in
Assignment 2, every request withholds the measured objects and traffic lights,
so the model finds them in the image. Your arbiter and your request policy
receive exact measurements: the scene as the simulator holds it, not an
estimate. The outcome points reward what you do with them, within the two
rules under [Rules](#rules).

Your observation is built once for a drive and called for every request, so it
may keep what its earlier requests carried, such as earlier frames or the car's
speed, and use it in a later one: to tell the model how the scene has changed,
or to build the one image the model receives from more than one frame. What it
keeps must come from its own requests, as the second rule under
[Rules](#rules) says.

### The values in `agent.yaml`

Assignment 3 permits every key
[Assignment 2 permits](../assignment-2/README.md#the-values-in-agentyaml), and
these:

| Keys | What they set | Allowed values |
| --- | --- | --- |
| `vla.minimum_interval_s` | the least time between two requests, in s | only raised |
| `vla.vertex.temperature` | how far Gemini's sampling varies; 0 asks for its most likely answer every time. Grading samples at yours | 0 to 2 |

The request budget, the endpoint, the model, and which scene fields a request
carries are the instructors', and no `agent.yaml` can change them.

## Practising on Vertex

Each team has its own Google Cloud project for practice.
[Model providers](../../docs/model-providers.md#vertex-course-provider) describes
how to install the cloud extra, sign in, and check the connection with
`cloud-check`. The project identifier the instructors give you, in
`GOOGLE_CLOUD_PROJECT`, is the only setting you set. Then drive the Vertex demo
with your submission, writing an event log:

```bash
uv run metadrive-starter run --config configs/demo-vla-vertex.yaml --submission submission --event-log tmp/vertex-01.jsonl
```

List the log in `EVENT_LOGS` in the [request timing](request-timing.ipynb)
notebook to see each request against the commands it bought. Event logs are
appended to, so give every run a new one.

Every request is billable. The project has a spending cap, which the
instructors can raise on request. The demo is a first connection, not a
measurement: one stopped car, at most 10 requests a run, and at most one every
2.5 s. It runs at half speed, so Gemini's delays take half as long in simulated
time as they do in grading, and its requests carry the measured objects, which
the practice scenes and the hidden drives withhold. Take your figures from the
practice scenes below.
Do your timing work in the notebook first, and confirm it on Vertex.

### The practice scenes

[`configs/practice-assignment-3/`](../../configs/practice-assignment-3/) holds
ten scenes and two faulted drives at the hidden drives' terms: the same
timing, real time, a request budget for each drive, and a request that
withholds the objects, so Gemini has to find them in the image. The roads and
the distances are not the hidden ones. Each file's first lines say what stands
on the road and what the drive must do.

| Scene | What stands on the road | Request budget |
| --- | --- | --- |
| `stopped-car` | one stopped car in your lane | 65 |
| `truck-and-parked-car` | a stopped truck in your lane, a parked car two lanes over | 100 |
| `two-cars` | a stopped car in your lane, another in the lane you pass it in | 105 |
| `road-closed` | barriers across every lane: stop and stay stopped | 105 |
| `open-road` | nothing, on two curves | 80 |
| `slow-truck` | a truck ahead at 3.5 m/s: follow or overtake | 65 |
| `roadworks` | cones and a barrier closing your lane, then the next lane | 90 |
| `mixed` | a stopped truck, a slow car, road works, and a stopped car | 110 |
| `late-car` | 300 m of empty road, then a stopped car in your lane | 20 |
| `late-roadworks` | 390 m of empty road, then road works and a stopped truck | 35 |
| `slow-truck-outage` | `slow-truck` with a 10 s provider outage on the approach | 65 |
| `road-closed-quota` | `road-closed` with quota refusals from 8 s to the end | 105 |

The two `late-` scenes have small budgets on purpose: a policy that asks on a
timer has none left when the obstacle comes into view, and without the model
nothing can change lane.

Drive one scene, or all twelve drives. An event log is appended to, so every
repetition needs a log of its own: change `practice-01` each time you repeat
the first command, and the second names its directory by the time it starts.

```bash
uv run metadrive-starter run --config configs/practice-assignment-3/stopped-car.yaml --submission submission --headless --event-log tmp/practice-01/stopped-car.jsonl
```

```bash
logs=tmp/practice-$(date +%m%d-%H%M%S); for scene in configs/practice-assignment-3/*.yaml; do uv run metadrive-starter run --config "$scene" --submission submission --headless --event-log "$logs/$(basename "$scene" .yaml).jsonl"; done
```

Each run ends by printing its summary: whether the car arrived or crashed,
`safety_interventions`, the control steps on which the safety supervisor
braked for you, and under `vla_metrics` the requests made and how many answers were accepted,
modified, or rejected. One pass of the twelve makes at most 945 requests, and a
request refused by a fault costs budget but is not billed. Gemini's answers
vary from run to run, so drive a scene several times before you trust a
figure: grading takes the median of three. These scenes are where your team's
own figures for the oral defence come from.

Plan for it: the scenes run at real time, so one pass takes up to 21 minutes of
driving, less where the car arrives early, and three passes up to 2,835
requests and about an hour, before each run's start-up.

`--headless` opens no window, but the camera frame each request carries is
still rendered, so these drives need working graphics:
[Platform notes](../../docs/running-and-configuring.md#platform-notes) has a
check, and what to do on a machine without a display. A machine that cannot
render at real time simulates more slowly than the scene asks, and every
answer then arrives younger in simulated time than it took, which flatters
your timing. The [request timing](request-timing.ipynb) notebook prints the
pace each log held: take no figure from a run that fell behind.

## Check your work

Every submission must pass all of Assignment 3's structural gates. They catch
what breaks a run, never how well it drives, and they drive your whole
submission on the Assignment 1 routes with a recorded answer, each under a
request budget, and under four provider faults:

```bash
uv run metadrive-starter gates --manifest configs/gates-assignment-3.yaml --config configs/demo-vla-fixture.yaml
```

| Gate | Fails when |
| --- | --- |
| submission loads | a file fails to import or lacks its entry point or methods, or `agent.yaml` sets a key or value it may not |
| bounded control outputs | a control action is not two finite numbers within `[-1, 1]` |
| no run stops on an error | a route stops on an error; a crash, an off-road exit, or an unfinished route fails no gate |
| valid observations emitted | an observation is not an `Observation` with an `RGBFrame`, non-empty text, and the captured timestamp |
| control loop never blocked | a `build` overruns the request timeout, 1 s here, or an arbiter or request policy call one control step |
| monotone restriction preserved | `arbitrate` grants more authority than the model proposed, `review` or `should_request` answers anything but True or False, or any of them raises |
| request budget respected | a drive attempted more requests than its budget |
| faults never break the run | a fault scenario stops on an error; it need not arrive |
| nothing outside `submission/` was modified | a file shipped outside `submission/`, notebooks aside, changed; this gate runs only in a student release |

`--scenario provider-outage` drives one scenario while iterating and marks the
report partial; before handing in, run them all.

## What you hand in

A zip of your `submission/` folder, uploaded to Moodle by the deadline, as
[Hand in](../../docs/getting-started.md#hand-in) describes: `request_policy.py`;
your `observation.py`, `arbitration.py`, and `controller.py`, all of which
drive in grading; and `agent.yaml`, with the values of all three assignments.
To check the zip, run the gates above on its unpacked copy.

### For the oral defence

There is no written report, so bring your own figures. Drive
[the practice scenes](#the-practice-scenes) with the submission you hand in,
each at least three times, and have for each scene:

- how many of its runs reached the end the scene declares;
- the requests made, against the scene's budget, and whether the budget ran
  out (`provider_requests_attempted` and `request_cap_exhausted` under
  `vla_metrics`);
- how many of Gemini's answers reached the car, as given or changed by your
  arbiter or the safety floor, and how many were refused
  (`validations_accepted`, `validations_modified`, and `validations_rejected`);
- `safety_interventions`, the control steps on which the safety supervisor
  braked for you.

It helps to have the same figures for the shipped `request_policy.py`, so that
what your policy changed can be seen. Be ready to explain every rule of your
policy: what it asks on, why, and where each value comes from.

## How you are marked

Assignment 3 is marked out of 100, in two parts. Hidden closed-loop drives on
Gemini compute points, and those points carry most of the mark. Staff then
review every submission, reading its code against those drives, and that
review is the rest of the mark: points the code does not account for are not
earned by the drives alone, so a number tuned against the public routes is not
a mark. In `request_policy.py` staff read what each rule asks on and why, and
where each value comes from. Write code a reader can follow: document each rule
and each value where it is written, and say what it is for. There is no written
report: your team explains and analyses its work at the oral defence.

Teams never produce a graded number: the notebook, the public routes, and your
own Vertex drives are for your own measurements. This marking does not change
once Assignment 3 is released: if a correction is ever needed, it is announced
and applies to every team alike.

### The floor: hidden structural gates

Grading runs [the structural gates](#check-your-work) on your `submission/`
against hidden routes, budgets, and fault timings, with the same command and
code you run on the public ones. If every gate passes, or reports "not run",
the assignment is marked from the hidden drives below; if any gate fails, the
assignment earns nothing. Passing the gates in public is a close reference, not
a guarantee: code that raises on a road or a fault timing it never met fails
there. You may replace your upload on Moodle until the deadline, and only the
last upload is graded; once the deadline closes there are no resubmissions. A
failure caused on the grading side is graded again by staff.

### The hidden drives

- **The model** is Gemini 3.5 Flash Lite through Vertex, asked from one
  instructor project for every team alike. That project has no spending cap,
  so grading never meets a quota of its own: the only refusals are the ones a
  scene schedules. It samples at your `vla.vertex.temperature`, or at 0 when
  your `agent.yaml` sets none.
- **Your whole submission drives**: your controller, observation, arbiter, and
  request policy, after your `agent.yaml`.
- **The scenes** are roads with obstacles, such as stopped cars and trucks,
  cones and barriers, in one lane or several, and a slower car ahead, and an
  empty road. Each declares the outcome it expects. They use other maps and
  distances than any public configuration.
- **Faults, mid-drive.** Some scenes with an obstacle are driven a second
  time with a fault: a 10 s provider outage, or quota refusals from some point
  to the end of the drive. When each begins is not published; the public gates and
  the practice scenes carry one of each kind, at their own times. A refused request costs budget, as
  [Between two requests](#between-two-requests) says, and once the command in
  force ends, the local fallback drives until an answer arrives again.
- **Timing.** Each command holds for 4.0 s from its request, an answer more
  than 2.5 s old is refused, DriveBench asks at most every 0.5 s, and a request
  times out after 6 s. Each drive has its own request budget, which your policy
  reads from `settings.request_budget`. On some scenes the obstacle comes after a
  long empty road and the budget is small, as on the practice scenes `late-car`
  and `late-roadworks`: requests spent while nothing changes are then missing
  when something does.
- **Three runs.** Every scored drive is driven three times, and each figure
  below uses the median of the three. A submission that is stable across runs
  does better than one that drove well once.

### The mark

Five measures earn 60 points between them, and the computed points are their
total scaled to 100, which the staff review above then completes. Waste,
reaction, and fault conduct count only on drives that reach the end their scene
declares. The Assignment 1 routes in the gate manifests check that nothing
breaks, and earn nothing.

#### Outcome: 40 points

A drive passes when it ends as its scene declares, with no collision and
without the car leaving the road at any point. Where there is a way past, the
car must reach the end of the scene: how it got past each obstacle is not
checked, and behind a slower car, following it and passing it both count. Where
every lane is blocked, the car must be at rest before the obstruction, slower
than 0.1 m/s, when the drive ends: how far short of it the car stopped is not
checked. Time is not scored.

Your share of the scored drives that pass is compared with the share the
shipped files reach on the same drives. Points: 40 × (your share − the shipped
files' share) ÷ (the share the instructors set from their reference solution's
drives − the shipped files' share), never below 0 and never above 40.

#### Waste: 6 points

A request is wasted when it is sent while the last answer's command still
holds, more than 2.5 s before its horizon ends, and nothing has changed since
that answer's request: the same objects within 50 m, the same lane clearance on
each side, the car's speed within 0.5 m/s, and no fault begun or ended. A
request in the last 2.5 s of a command is never wasted: its answer arrives
about when the command ends. During a fault, a request sent less than 5 s after
the one before it is wasted, whatever came of it.

A drive's waste share is its wasted requests over all its requests. Points:
6 × (1 − the median waste share over the scored drives).

#### Reaction: 6 points

An event is an object coming within the measured 50 m in your lane or a lane
beside it, or a lane beside the car changing between clear and not clear. Each
event scores by the delay from it to your next request: 1 at 1.0 s or less,
falling evenly to 0 at 3.0 s. A drive's figure is the mean over its events;
drives without an event do not count. Points: 6 × the median over the drives.

#### Fault conduct: 6 points

On each drive with an outage, the delay from the end of the outage to your next
request scores 1 at 3.0 s or less, falling evenly to 0 at 10 s. Asking through a
fault is counted in waste, not here: a policy that asks once every 5 s during a
fault wastes nothing, and finds the provider back within 5 s. Points: 6 × the
median over the drives with an outage.

#### Syntax: 2 points

The share of the answers that decode, over every answer that arrives on any
scored drive. An answer the safety floor refuses for its age or its confidence
still decoded. Points: 2 × the share.

#### Scene interpretation: no points

The grade report gives how well the hazards in Gemini's answers match the
measured scene, and staff read it beside your code. Each
vehicle, pedestrian, obstacle, or traffic control an answer reports is a claim,
paired with the measured scene as
`metadrive_starter.hazard_agreement.score_hazards` pairs them, as in
[Assignment 2](../assignment-2/README.md#scene-interpretation-15-points). A drive
records nothing about what the camera showed, so **recall** counts only the
measured objects within 50 m in the three front sectors, `front`, `front_left`,
and `front_right`; **precision** counts every claim. Each run pools every
answer of every scored drive into one F1.

Guessing earns an F1 too, so the report sets your F1 against the F1 of the best
answer that is the same on every request, scored on the same requests, and
against the F1 the instructors set from their reference solution's drives. It
earns no points of its own: what your observation does for a drive is paid in
the outcome.

The grade report gives two more figures that earn no points: the share of the
model's answers that reached the car as the model gave them, and the share that
the instructors' own arbiter and the safety floor would have let stand.

### The two rules, checked

The two rules under [Rules](#rules) keep the model in the loop. Grading checks
both. It builds every prompt of your drives again in a fresh process that holds
your `observation.py` alone, which shows any prompt built from something its
request did not carry. And it asks the model about recorded scenes with each
scene's image swapped for another's, which shows whether the answers follow the
image or the prompt. Staff read the code of a submission either check flags
before its drives count.

## Rules

- Change only files in `submission/`. The simulator, the model providers, the
  safety floor, and the gates are course code that your files call but never
  edit.
- Import only what the course environment already provides: the Python standard
  library, `metadrive_starter`, and its installed dependencies. Grading
  installs nothing of yours.
- Keep every tuned value that has a [permitted key](#the-values-in-agentyaml)
  in `agent.yaml`, and any other, such as how long your policy waits after a
  failure, as a named constant at the top of the file that uses it, where a
  reader finds it. Your files read nothing but what DriveBench passes them: no
  other files, no environment variables, no network. Your request policy only
  decides; it never asks a model, and neither does your arbiter.
- **Your prompt puts the question to the model.** It may tell the model what
  to look for and how to answer, but it may not dictate the answer or the
  hazards: no prompt that works out the action, or names what the scene holds,
  and has the model repeat it.
- **No seam file passes the measured scene to `observation.py`.** Your arbiter
  and request policy receive the whole measured scene; your observation
  receives only its request, which withholds the objects and traffic lights.
  Nothing your files keep, such as a module-level variable another file reads,
  may carry what the request withholds into a prompt.
- Measure on the notebook's road, the public routes, the Vertex demo, the
  practice scenes, or maps and seeds you choose yourself. Hidden scenes are driven only by the
  instructors.

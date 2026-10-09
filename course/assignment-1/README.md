# Assignment 1: PID control

[← DriveBench README](../../README.md) · [Rubric](rubric.md) · [PID lab notebook](pid-lab.ipynb)

Run all commands from the repository root.

Assignment 1 carries 15% of the assignment marks, which together make up 60% of
the project grade; the oral defence at the end of the project makes up the
other 40%. The course calendar gives the deadline, and late submissions are not
accepted.

## Why this assignment comes first

In later assignments a vision-language model assesses each scene and proposes
what the car should do. It never actuates the vehicle: local code decides
whether to act on the proposal, sets the target speed, and follows the route.
Your controller is the last step of that chain: every throttle, brake, and
steering command the vehicle ever receives comes out of it. Assignment 1 builds
it while nothing else is uncertain, and gets your team through installing,
running, measuring, and handing in before any model enters the course.

## What you hand in

The contents of `submission/`, as your instructors direct, and nothing else:

| File | What it holds |
| --- | --- |
| `controller.py` | your controller: a complete PID in each loop |
| `agent.yaml` | the gains and planner values you tuned |

Never edit a file outside `submission/`. Grading places your `submission/` in a
clean copy of the course repository, so a change anywhere else never reaches
the grader, and the structural gates flag it before you hand in.

## Set up

Follow [Local Setup](../../README.md#local-setup) in the README, then confirm
that the shipped files drive:

```bash
uv run metadrive-starter smoke --submission submission
uv run metadrive-starter gates
```

The gates should report `PASSED` with a score of about 8. The shipped
controller completes every route, badly: see [What the stub does](#what-the-stub-does).

## Your controller

[`submission/controller.py`](../../submission/controller.py) must define `Controller(settings)`. The simulator
builds one per run and calls it on every 0.1 s control tick:

- `update(tick)` returns `(steering, throttle_brake)`, two finite numbers
  within `[-1, 1]`. A negative `throttle_brake` brakes, and a positive
  `steering` steers left.
- `reset_speed_control()` must make the speed loop forget its history, exactly
  as if it had just been built. The simulator calls it when the target speed
  jumps by at least `controller.speed_pid_reset_threshold_mps`, when the source
  of the target speed changes, and on every tick the safety floor overrides the
  throttle or brake.

Each `tick` carries `target_speed_mps`, `speed_mps`, `heading_error_rad`,
`lateral_error_m`, and `dt_s`, the tick length in seconds. Both errors are
signed so that a positive error calls for positive (left) steering.

`settings` holds the `controller` section of the configuration, with your
`agent.yaml` applied: `target_speed_mps`, `speed_pid_reset_threshold_mps`, and
three loops, `speed_pid`, `steering_pid`, and `lateral_pid`. Each loop has
`kp`, `ki`, `kd`, `output_min`, `output_max`, and `anti_windup`.

**Take every gain from `settings`.** Your controller must be built from the
values it is given, never from numbers written into the code. The
implementation checks below construct it with gains of their own.

## The task

1. **Complete the PID.** The shipped `PID` class implements only the
   proportional term. Add:
   - the integral term, integrating the error over seconds, so it uses `dt`;
   - the derivative term, as the rate of change of the error per second;
   - anti-windup, so that the integral cannot keep growing while the output is
     saturated at `output_min` or `output_max`. Let `anti_windup: false` switch
     it off: the PID lab uses that to show windup, though no check grades it;
   - `reset()`, clearing everything the loop remembers.

   Every loop uses the same class, so the steering loops gain the same terms.
2. **Remove the handicap.** `Controller.__init__` overrides the speed gain with
   a deliberately weak `kp=0.03`. Delete that override.
3. **Check each term on its own** with the [PID lab](#the-pid-lab) and the
   implementation checks in `uv run metadrive-starter gates`.
4. **Tune** gains and planner values in [`submission/agent.yaml`](../../submission/agent.yaml), measuring every
   change on the public routes.
5. **Keep a record** of what you try and measure, with the command behind each
   figure: there is no written report, but at the oral defence your team
   explains and analyses its work.

You may restructure `controller.py` however you like, as long as it keeps the
`Controller` contract above and reads its gains from `settings`. Run
`uv run metadrive-starter smoke --submission submission` after every change: it
loads your files and computes one control tick without starting the simulator.

## The PID lab

[`pid-lab.ipynb`](pid-lab.ipynb) is where you write and tune your controller. It
holds a copy of the shipped controller and of every loop's gains, and walks you
through the assignment in six tasks, one for each `TODO` in the controller:
remove the weak speed gain, add the integral term, the derivative term,
anti-windup, and the reset, and tune the gains. Tasks 1 to 5 drive a toy vehicle
and end with the gate check that grades them; a checklist shows all six
implementation checks at once. Task 6 drives the public routes in MetaDrive with
your gains and prints what the gate report prints for each. It needs the optional
notebook packages, which the command below installs:

```bash
uv run --extra notebooks jupyter lab course/assignment-1/pid-lab.ipynb
```

The gates and grading never read the notebook, only `submission/`. When you are
happy, copy the code in **Your controller** into
[`submission/controller.py`](../../submission/controller.py), below its
docstring, and the gains into [`submission/agent.yaml`](../../submission/agent.yaml), then run the notebook's
last cell: it checks that your submission drives exactly like the notebook. Set
`DRIVE = "submission"` in **Your gains** to run every experiment on your
submitted controller instead.

Without a browser, run it headlessly and open the executed copy it writes to
`tmp/`:

```bash
uv run --extra notebooks jupyter nbconvert --to notebook --execute course/assignment-1/pid-lab.ipynb --output-dir tmp
```

The notebook is never submitted or graded. Edit and save it as you like: running
or changing it does not count as modifying a file outside `submission/`, and a
course release never changes it once shipped, so your edits survive every pull.
Saving it with its outputs is fine, though a course test that checks notebooks
ship without outputs then fails on your machine; that test is for course
maintainers, and you can ignore it.

## Tuning

The starting gains in [`configs/pid.yaml`](../../configs/pid.yaml) drive every route, but they are only a safe
starting point, not tuned: a complete PID on them earns about 55 of the driving
credit on the public routes and 63 on the hidden ones, because full credit asks
for closer speed and lane tracking than they achieve. Tuning earns that credit,
and it involves trade-offs: a stiffer
loop can track more closely and cost time or speed error elsewhere, and a loop
too soft or too stiff can leave the road.

[`submission/agent.yaml`](../../submission/agent.yaml) is an overlay, not a configuration: write only the keys
you change, nested as in [`configs/default.yaml`](../../configs/default.yaml), or in [`configs/pid.yaml`](../../configs/pid.yaml) for
the gains. Assignment 1 permits:

| Keys | Allowed values |
| --- | --- |
| `kp`, `ki`, `kd` of `controller.speed_pid`, `controller.steering_pid`, `controller.lateral_pid` | at least 0 |
| `anti_windup` of each loop | `true` or `false` |
| `output_min`, `output_max` of each loop | within `[-1, 1]` |
| `controller.speed_pid_reset_threshold_mps`, `planner.curvature_preview_m` | any valid value |
| `controller.target_speed_mps`, `planner.maximum_lateral_acceleration_mps2`, `planner.minimum_curve_speed_mps` | only lower than the instructor value |

Any other key, or a value outside its bounds, is rejected by name. The file
also accepts Assignment 2's keys, `vla.prompt_policy` and the `observation`
section, which change nothing in Assignment 1's runs. Print what your runs will
use with `uv run metadrive-starter config --submission submission`.

Useful measurements:

```bash
uv run metadrive-starter gates
uv run metadrive-starter gates --json
uv run metadrive-starter run --submission submission --headless --traffic-density 0 --map C
uv run metadrive-starter run --submission submission --render --traffic-density 0 --map XTOC
```

`gates` drives all five public routes and prints every gate, implementation
check, and driving figure, rounded to two decimals; `--json` gives every figure
at full precision, which small changes need, and adds each file's SHA-256.
`run` drives one route; its JSON summary gives `steps`, `arrived`, `crashed`,
and `went_off_road`, and with `--render` it opens a window: the default profile
runs headless. For evidence the summaries do not give, such as overshoot or
time at full throttle, add
`--event-log tmp/run.jsonl` to `run`: it writes one `control_applied` event per
control tick, with the vehicle's state, the target speed, and the action
applied. The default profile has light traffic,
so pass `--traffic-density 0` to drive as the gates do. Runs are deterministic:
the same files and command give the same numbers. The planner caps the target
speed before curves, so the target your speed loop receives changes along
every route.

The public routes all use seed 0, and `run` has no seed option. To drive other
seeds, copy [`configs/default.yaml`](../../configs/default.yaml) to a file of your own, such as
`tmp/my-routes.yaml`; set `simulator.map`, `simulator.start_seed`, and
`simulator.traffic_density: 0.0`; and pass it with `--config`. To run the
gates and the score on routes of your choosing, copy
[`configs/gates-assignment-1.yaml`](../../configs/gates-assignment-1.yaml), change its maps and seeds, and pass it with
`gates --manifest`. Only the reference defines par, so set each `par_steps` to
your own controller's steps on that route.

### Measuring an attempt you do not keep

Every figure you rely on should be one anyone can rerun, including figures
from attempts you did not hand in. Measure each attempt in a copy of your
submission, never by editing and reverting `submission/` itself:

```bash
mkdir -p tmp && cp -R submission tmp/attempt-1
uv run metadrive-starter gates --submission tmp/attempt-1 --json
```

Edit `tmp/attempt-1/agent.yaml` or `tmp/attempt-1/controller.py` between the
two commands. `gates`, `run`, and `config` all take `--submission` with any
directory. Record the attempt's complete `agent.yaml`, or its exact code
change, beside the command, so that anyone can rebuild it. The same way, a copy with a term of your controller removed
shows what that term does in the implementation checks.

## Acceptance thresholds

Every submission must pass all four structural gates. They run on the five
public routes when you run them, and on hidden routes at grading:

| Gate | Passes when |
| --- | --- |
| submission loads | `controller.py` imports, defines `Controller(settings)` with `update(tick)` and `reset_speed_control()`, and `agent.yaml` sets only permitted keys within their bounds |
| bounded control outputs | every action is two finite numbers and never needs clamping into `[-1, 1]` |
| no run stops on an error | every route is driven to its end without an exception; a route that crashes, leaves the road, or does not arrive within its 2000-step horizon fails no gate, but earns no [driving credit](#driving) |
| nothing outside `submission/` was modified | every file shipped outside `submission/`, notebooks aside, is unchanged; this gate runs only in a student release |

The hidden routes have no traffic, like the public ones. They are built from
the same kinds of road — straights, curves, intersections, T-junctions, and
roundabouts — on maps the public manifest never uses. Try other maps yourself
before handing in, for example `--map T` or `--map OX`: gains that hold every
public route can still leave the road on one you have never driven, and that
route then earns no driving credit.

A gate that fails at grading is one you could have run: the grader uses the
same command and the same code, only a different route list.

## The score

The gate run also prints a score out of 100. It never decides a gate. Half of
it comes from the implementation checks, half from how your car drove the
routes. You see the score on the public routes; grading uses the score on the
hidden routes.

### Implementation checks

MetaDrive's car keeps its speed with almost no throttle, so on the routes a
proportional-only controller drives about as well as a complete PID. The
implementation checks test each term directly instead, on a toy vehicle that
accelerates and brakes like MetaDrive's car, and on hills it never has. Each
check builds your `Controller` with its own gains, so it tests your
implementation, not your tuning. Each passing check is worth one sixth of the
implementation half.

| Check | Passes when |
| --- | --- |
| speed loop integral term | with `speed_pid` `kp = 0.2`, setting `ki = 0.2` adds `ki` × error × time to `throttle_brake`, within 10%, after 2 s at a steady 0.5 m/s error, with 0.1 s and 0.05 s ticks |
| speed loop derivative term | `kd = 0.1` adds `kd` × the error's rate of change per second, within 10%, while the error changes by 1 m/s every second, with both tick lengths |
| steering loops' integral and derivative terms | `steering_pid` and `lateral_pid` each apply `ki` and `kd` the same way to `steering` |
| speed held against a constant load | with `kp = 0.3`, `ki = 0.1`, cruising at a 10 m/s target onto a 0.5 m/s² slope, the speed is back within 0.05 m/s of the target after 40 s |
| no windup after a saturated start | the same gains, from standstill up the same slope, overshoot the target by at most 1 m/s and are within 0.05 m/s of it after 40 s |
| `reset_speed_control()` clears speed history | with `kp = 0.3`, `ki = 0.1`, `kd = 0.1`, after 5 s at a steady 0.5 m/s error and a reset, `throttle_brake` matches a freshly built controller's tick for tick; a loop that keeps no history has nothing to clear and fails |

A term's contribution is measured as the difference between two controllers
that differ only in the gain under test, so a feedforward term of your own
cancels out, and a derivative filter only has to settle within the first
second. A check that fails says what it measured and what it expected.

### Driving

Each route earns driving credit from 0 to 1, from three measurements:

| Measurement | Weight | Full credit | No credit |
| --- | --- | --- | --- |
| steps to arrival, as a multiple of the route's par | 0.4 | par or faster | 1.3 × par or slower |
| mean \|target speed − speed\| after the first 5 s | 0.4 | 0.1 m/s or less | 0.6 m/s or more |
| root mean square lateral error over the route | 0.2 | 0.02 m or less | 0.15 m or more |

Credit falls linearly between the two columns. Par is the steps the instructor
reference controller takes on that route; the public manifest,
[`configs/gates-assignment-1.yaml`](../../configs/gates-assignment-1.yaml), lists it as `par_steps`. Finishing faster
than par earns nothing more. A route that crashes, leaves the road, or does not
arrive earns no credit, and the report says why. The driving half is the mean
over all routes.

The reference, a complete PID on gains the instructors tuned from the starting
ones, earns about 80 of the driving credit on the public routes and 85 on the
hidden ones: its time is par, but its speed and lateral errors sit between the
two columns.

The speed error is measured against the target your controller is given, after
the planner's curve limits. Lowering your target speed makes that error easier
to keep small, and costs more time credit than it saves.

### What the stub does

The shipped controller passes every gate and scores about 6 on the public
routes: no implementation check passes, and its weak speed gain keeps it 1.7 to
3.5 m/s below its target and 1.2 to 1.6 times slower than par. Removing the
handicap alone brings the score to about 26: on the starting gains,
proportional-only control drives these routes about as well as a complete PID. A
complete PID on the starting gains scores about 77, and tuning earns the rest of
the driving credit.

## How it is graded

Assignment 1 is marked automatically: the hidden structural gates are a floor,
and the score on the hidden routes, out of 100, is the mark. There is no written
report. The [rubric](rubric.md) sets out how the mark is computed.

## Rules

- Change only files in `submission/`. The simulator, the planner, the safety
  floor, and the gates are course code that your controller calls but never
  edits.
- Your controller may import only what the course environment already
  provides: the Python standard library, `metadrive_starter`, and its installed
  dependencies. Grading installs nothing of yours, so any other import fails the
  "submission loads" gate.
- Keep every tuned value in `agent.yaml`. The controller reads nothing but its
  `settings`: no other files, no environment variables, no network.
- Measure on the public routes, or on maps and seeds you choose yourself, as in
  [Tuning](#tuning). Hidden routes are run only by the instructors.

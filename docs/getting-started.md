# Getting started

[← DriveBench README](../README.md) · [Next: VLA pipeline and authority →](vla-pipeline.md)

Run all commands from the repository root unless stated otherwise.

## Get the code and stay up to date

The course publishes DriveBench as a git repository, and every assignment
arrives as a new release on it. Your team works in a private repository of its
own and pulls each release from the course repository, kept as a remote called
`upstream`. Your work lives in your repository; the course repository only ever
brings updates.

One team member sets this up once. Create an empty private repository for the
team on your git host, with no README, licence, or other files, then:

```bash
git clone https://github.com/sntubix/cav-drivebench.git drivebench
cd drivebench
git remote rename origin upstream
git remote add origin <your team's repository URL>
git push -u origin main
```

Everyone else on the team clones the team's repository and adds the same
`upstream`:

```bash
git clone <your team's repository URL> drivebench
cd drivebench
git remote add upstream https://github.com/sntubix/cav-drivebench.git
```

Commit your work and push it to `origin`, your team's repository, as usual.
Never push to `upstream`: it belongs to the course.

When the course announces a release, pull it into your work, then push the
result to your team's repository:

```bash
git pull --no-rebase --no-edit upstream main
git push
```

A release changes only files outside `submission/`, and adds new stubs inside it
as new files, such as the next assignment's. It never changes a file in
`submission/`, or a course notebook, that has already shipped. Every assignment
asks you to change only `submission/`, and the notebooks are yours to edit, so
the pull merges without a conflict. If git reports one anyway, it is in another
file you changed: take the course's version of that file, then finish the merge:

```bash
git checkout upstream/main -- <file>
git commit --no-edit
```

Set up your repository as above rather than with your git host's "fork" or
"use this template". A template copy shares no history with the course
repository, so releases no longer merge into it, and a fork of a public
repository is public too.

## Prerequisites

DriveBench supports Python `>=3.10,<3.12` because MetaDrive `0.4.3` does not
support Python 3.12. Install [`uv`](https://docs.astral.sh/uv/) if it is not
already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment, including the development checks:

```bash
uv sync --group dev
```

## Verify the installation

A student release has no controller but the one in `submission/`, so pass
`--submission submission` to every `smoke`, and to every `run` in this guide
that is not under `--manual` control; without it, the command stops and asks
for one.

First validate configuration and application wiring without launching MetaDrive:

```bash
uv run metadrive-starter smoke --dry-run --submission submission
```

Then launch the deterministic, platform-tested autopilot profile. The first
launch downloads MetaDrive's assets, which takes a minute or more:

```bash
uv run metadrive-starter run \
  --config configs/demo-autopilot.yaml \
  --submission submission \
  --headless \
  --steps 1000
```

The summary should report `arrived: true`, `crashed: false`, and
`went_off_road: false`.

Runs are unpaced by default, which is useful for local simulation and tests.
`--realtime` prevents one simulation second from passing faster than one wall-clock
second. `--realtime-factor 0.5` advances at half real time, giving a slower model
two wall-clock seconds per simulation second. `--unpaced` overrides a paced YAML
profile.

## Your submission

Team code lives only in `submission/`; never edit files outside it. The
[Assignment 1 brief](../course/assignment-1/README.md) sets the first task, its
thresholds, and its rubric. Pass the directory to `run` or `smoke` and your
controller drives. A student release has no other controller, so a run without
`--submission` stops and asks for one:

```bash
uv run metadrive-starter smoke --submission submission
uv run metadrive-starter run --submission submission --headless --map C
```

`smoke` loads your code and computes one control tick without launching
MetaDrive, so run it first after every change. The summary's `controller` field
names what drove: `submission.controller.Controller` is yours.

[`submission/controller.py`](../submission/controller.py) must define `Controller(settings)`. `settings` holds
the `controller` values from configuration, and the object it builds must
provide two methods:

- `update(tick)` returns `(steering, throttle_brake)`, both finite and within
  `[-1, 1]`; a negative `throttle_brake` brakes. `tick` carries
  `target_speed_mps`, `speed_mps`, `heading_error_rad`, `lateral_error_m`, and
  `dt_s`. Both errors are signed so that a positive error calls for positive
  (left) steering.
- `reset_speed_control()` clears speed-control history. It is called when the
  target speed jumps by at least `controller.speed_pid_reset_threshold_mps`,
  when the source of the target speed changes (for example from a validated
  command to the local fallback), and on every tick the safety floor overrides
  throttle or brake.

[`submission/agent.yaml`](../submission/agent.yaml) tunes permitted values: those your `Controller`
receives in `settings`, and a few that shape the route it tracks. It is an
overlay, not a full configuration: write only the keys you change, nested as in
[`configs/default.yaml`](../configs/default.yaml), or in [`configs/pid.yaml`](../configs/pid.yaml) for the gains. Assignment 1 permits:

- under `controller.speed_pid`, `controller.steering_pid`, and
  `controller.lateral_pid`: `kp`, `ki`, and `kd` (at least 0), `anti_windup`,
  and `output_min` and `output_max` (within `[-1, 1]`);
- `controller.speed_pid_reset_threshold_mps` and
  `planner.curvature_preview_m`;
- `controller.target_speed_mps`, `planner.maximum_lateral_acceleration_mps2`,
  and `planner.minimum_curve_speed_mps`, which may only be lowered.

Later assignments permit more keys, which their briefs list.
Any other key, or a value outside its bounds, is rejected by name. Print the
configuration your runs will use with
`uv run metadrive-starter config --submission submission`.

A file that cannot be loaded stops the run before MetaDrive starts, with a
message naming what was expected. An action outside `[-1, 1]` is clamped, as a
real actuator saturates, and counted in the summary's
`controller_outputs_clamped`; an action that is not two finite numbers stops
the run at the tick that produced it.

Before handing in, run the structural gates:

```bash
uv run metadrive-starter gates
```

They check that your submission loads, that no control output needed
clamping, that no run stops on an error, and, in a student release, that
nothing outside `submission/` changed. A route that crashes, leaves the road, or
does not arrive fails no gate in Assignment 1: it earns no driving credit. Each
gate reports on its own, so one broken piece never hides another. Grading runs
the same gates on hidden scenarios, so a gate that fails there is one you could
have run yourself. While iterating, `--scenario straight` drives just that route
(repeat it to pick several) and marks the report partial; before handing in, run
them all.

The same run prints a score out of 100, which never decides a gate: half from
implementation checks that test each term of your controller on a toy vehicle,
half from how your car drove the routes. [The score](../course/assignment-1/README.md#the-score)
in the brief defines it, and graded scores come only from hidden scenarios.

The shipped `controller.py` is deliberately mediocre: every loop is
proportional-only and its speed gain is weak. It completes a route without a
crash or off-road exit, well below the target speed and far slower than the
reference.

Assignment 2 adds `observation.py`, what the model receives, and
`arbitration.py`, how far to trust what it answers. Both load only when a run
asks the model, and `agent.yaml` gains `vla.prompt_policy`, the `observation`
toggles, and the safety floor's `command_validation` thresholds, which may only
be tightened. Its brief, released with Assignment 2, sets out the task, the
keys, the gates, and the rubric.

## First demonstrations

Run the same waypoint-planning demonstration with your controller in a
rendered window:

```bash
uv run metadrive-starter run \
  --config configs/demo-autopilot.yaml \
  --submission submission
```

The display reports target speed, measured speed, steering, and throttle/brake.
The terminal reports route completion, arrival, collision, and off-road status.

Run the straight-road emergency-braking demonstration:

```bash
uv run metadrive-starter run \
  --config configs/demo-emergency-braking.yaml \
  --submission submission
```

This profile uses the `XTOC` map and places a static vehicle 25 m ahead. The PID
controller proposes ordinary controls while the local safety supervisor may
remove throttle or apply service/emergency braking.

<details>
<summary>More rendered, curved-road, and manual demonstrations</summary>

Open a rendered window:

```bash
uv run metadrive-starter run --submission submission --render --steps 1000
```

Run the curved-path emergency-braking profile:

```bash
uv run metadrive-starter run \
  --config configs/demo-emergency-braking-curve.yaml \
  --submission submission
```

The curved profile uses MetaDrive's `C` map and places a stopped vehicle 100 m
along the selected route, after the bend begins. Add `--headless` for a
non-visual run.

Run with keyboard control:

```bash
uv run metadrive-starter run --render --manual --steps 1000
```

Equivalent reusable profile:

```bash
uv run metadrive-starter run \
  --config configs/demo-manual.yaml \
  --manual
```

Use `W` to accelerate, `A`/`D` to steer, and `S` to brake and then reverse.
Reverse is enabled automatically in manual mode; autonomous runs retain
MetaDrive's brake-only negative action behaviour.

</details>

## Platform notes

The same local `uv` workflow targets Linux and macOS. Linux is supported; Docker
is its reproducible headless baseline. Native rendering has also been exercised
on Ubuntu and Apple Silicon, while supported local-model combinations are listed
separately in the [model-provider guide](model-providers.md).

On macOS, MetaDrive/Panda3D rendering can be more fragile than headless mode,
especially on Apple Silicon. DriveBench applies a macOS-only compatibility shim:
it caps unsupported 8x/16x MSAA requests at 4x, requests a core OpenGL context,
and adapts the mixed-generation shaders used by terrain, roads, the skybox, and
offscreen camera tonemapping.

## Docker

Build the simulator image:

```bash
docker compose build
```

Run a headless simulation:

```bash
docker compose run --rm metadrive \
  metadrive-starter run --submission submission --headless --steps 100
```

On Linux, run the complete automated Docker baseline from a fresh checkout:

```bash
./scripts/platform-check.sh
```

The validator builds the image, checks the frozen lock and bundled MetaDrive
assets, runs the tests, smoke check, and real route, captures the RGB probe
catalog under Xvfb, then, once a release includes Assignment 2's fixtures, runs
and replays the fixture-backed VLA loop. It writes new evidence under
`artifacts/platform-validation/<UTC timestamp>/` and never overwrites an earlier
run. Use `--skip-build` only while iterating against an image built from the
same checkout.

The automated command deliberately records visual review, GUI/input, live local
VLM, and live cloud VLM as separate gates. A successful exit means the automated
Docker slice passed; it is not evidence that every optional runtime was exercised.

For the native Ubuntu plus local-model gate, use the immutable downloads and
single command in the
[Linux llama.cpp acceptance recipe](model-providers.md#linux-local-server-acceptance-llamacpp).
That validator includes the native simulator baseline and requires two clean
model-server rounds with actual VLM authority and exact replay.

<details>
<summary>Rendered Docker execution</summary>

Rendered Docker execution requires host display forwarding. On Linux, a typical
X11 invocation is:

```bash
xhost +local:docker
docker compose run --rm -e DISPLAY=$DISPLAY metadrive \
  metadrive-starter run --render --manual
xhost -local:docker
```

The final command revokes the temporary local-container X11 permission after the
simulator exits.

On macOS, prefer local `uv` execution for rendered/manual mode unless a separate
display-forwarding setup is already available.

</details>

## Configuration

Defaults live in [`configs/default.yaml`](../configs/default.yaml), and the starting PID gains every profile
drives with live only in [`configs/pid.yaml`](../configs/pid.yaml). Select any profile with `--config`:

```bash
uv run metadrive-starter run \
  --config configs/default.yaml \
  --submission submission \
  --headless \
  --steps 500
```

Controller gains, target speed, route sampling, lookahead, curvature preview,
lateral-acceleration budget, perception, safety, and provider settings all live
in YAML; ordinary tuning does not require Python edits.

Profiles choose the world you experiment in. To try another scenario, pick a
profile with `--config`, override it with flags such as `--map` and
`--traffic-density`, or copy the closest demonstration profile to a new file.
The values your team is graded on do not go in a profile: they belong in
[`submission/agent.yaml`](../submission/agent.yaml) (see [Your submission](#your-submission)), which
applies on top of whichever profile you run.

All runtime/domain speeds use m/s: configuration, ego state, local scene, VLA
command, validation, planning, PID, probe metadata, event logs, and summaries.
Non-display `*_kmh` fields are rejected. Rendered display defaults to m/s; use
`--speed-unit kph` or `display.speed_unit: kph` when desired.

Set the local safety scene source explicitly:

```yaml
perception:
  safety_source: lidar  # or: oracle
```

`oracle` observes all simulator objects in range and is the deterministic
ground-truth baseline. `lidar` consumes only MetaDrive's detected-object set.
MetaDrive still supplies exact pose, dimensions, and velocity for detected
objects; raw point-cloud clustering and tracking are later perception work.

The planner projects ego position onto the selected route and keeps progress
monotonic. Curvature preview caps the local PID speed objective before tight
turns. The VLM chooses semantic behaviour, never direct steering.

<details>
<summary>Advanced configuration and command-line overrides</summary>

Select a map and traffic density without editing YAML:

```bash
uv run metadrive-starter run \
  --render \
  --manual \
  --traffic-density 0.25 \
  --map XTOC
```

Keep driving after leaving the road or colliding and enable procedural obstacles:

```bash
uv run metadrive-starter run \
  --render \
  --manual \
  --continue-off-road \
  --continue-after-collision \
  --obstacle-probability 0.2
```

`--continue-after-crash` aliases `--continue-after-collision`.
`--accident-probability` aliases `--obstacle-probability`. Obstacle probability
is applied independently to each eligible straight, curve, or ramp block.

Print the effective configuration:

```bash
uv run metadrive-starter config --render --manual
```

`planner.route_source: map` samples MetaDrive's selected route;
`planner.route_source: waypoints` uses `planner.default_route` from YAML. The
default `XTOC` map contains an intersection (`X`), T-junction (`T`), roundabout
(`O`), and curve (`C`). Other block identifiers include straight (`S`) and ramps
(`R`/`r`).

</details>

## Where to go next

Continue with [VLA pipeline and authority](vla-pipeline.md) to understand how
model assessments become locally bounded commands.

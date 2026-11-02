# Running and configuring DriveBench

[← Getting started](getting-started.md) · [DriveBench README](../README.md)

Reference for what [Getting started](getting-started.md) leaves out: rendered
demonstrations, pacing, platforms, Docker, and configuration. No assignment
needs any of it to get started. Run all commands from the repository root
unless stated otherwise.

## First demonstrations

Run the autopilot profile that [Getting started](getting-started.md#check-that-it-works)
runs headless, with your controller, in a rendered window:

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
controller proposes ordinary controls while the safety floor may remove
throttle or apply service/emergency braking.

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
catalog under Xvfb, then runs and replays the fixture-backed VLA loop. It writes new evidence under
`artifacts/platform-validation/<UTC timestamp>/` and never overwrites an earlier
run. Use `--skip-build` only while iterating against an image built from the
same checkout.

The automated command deliberately records visual review, GUI/input, live local
VLM, and live cloud VLM as separate gates. A successful exit means the automated
Docker slice passed; it is not evidence that every optional runtime was exercised.

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
[`submission/agent.yaml`](../submission/agent.yaml) (see [Your submission](getting-started.md#your-submission)), which
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

## Pacing

Runs are unpaced by default, which is useful for local simulation and tests.
`--realtime` prevents one simulation second from passing faster than one wall-clock
second. `--realtime-factor 0.5` advances at half real time, giving a slower model
two wall-clock seconds per simulation second. `--unpaced` overrides a paced YAML
profile.

## Files you generate

Keep what you generate under `tmp/`, which git ignores; the platform validators
write their evidence under `artifacts/`. Never commit credentials, unreviewed
model responses, or private hidden-evaluation output.

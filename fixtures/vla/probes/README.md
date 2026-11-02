# Recorded VLA probe scenes

These catalogs hold exact, model-independent probe inputs: the prompt that
DriveBench's original observation built, and the camera PNG, for scenes that
the Assignment 2 notebooks and the probe replay without MetaDrive. They contain
no credentials, project IDs, provider response IDs, or cloud configuration. The
frames ship, and no provider fixture is matched to them, so they serve on any
machine. Do not edit or rerender a catalog: the files beside each frame hold
its hashes, and a changed frame no longer replays.

## The eight public probe scenes

`catalog-20261002-v5/` holds the eight public probe scenes, captured with
[`configs/default.yaml`](../../../configs/default.yaml) and DriveBench's original observation. Each
scene directory is ordinary probe output plus `local_scene.json`: the
`LocalScene` measured with the frame, in the form event logs record it.
`metadrive_starter.recorded_probes` loads them and rebuilds each scene's
`ObservationRequest` exactly as `probe --submission` builds it, so the
Assignment 2 notebooks run your `observation.py` on these scenes without
MetaDrive.

They also replay without MetaDrive, sending the saved original observation,
for example to score a semantic-failure fixture on every scene:

```bash
uv run metadrive-starter probe \
  --config configs/demo-vla-semantic-partial-occlusion.yaml \
  --infer \
  --replay-from fixtures/vla/probes/catalog-20261002-v5 \
  --output-dir tmp/semantic-partial-occlusion-replay-01
```

## The public interpretation scenes

`interpretation-public-20261030/` holds eight scenes with pedestrians, cones, and
barriers, which the probe cannot place, at several distances and in each lane,
on maps and seeds of their own. Each scene directory is ordinary probe output
plus `local_scene.json`, recorded with
[`configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml`](../../../configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml),
so its saved prompt holds no object and no lane clearance, and two files of truth
that `score_hazards` scores against:

- `truth.json`: each placed object, located as hazard agreement scores it, with
  whether the camera shows it and whether it is within the measured radius;
- `truth_scene.json`: the scene measured to 80 m with the frame, so that a claim
  of an object past the 50 m radius pairs with it.

They replay in id order, through your observation:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-qwen3-vl-2b-llama-interpretation.yaml \
  --submission submission \
  --infer \
  --seed 1 \
  --replay-from fixtures/vla/probes/interpretation-public-20261030 \
  --output-dir tmp/probe-interpretation-01
```

## The Vertex capture catalog

`vertex-gemini-3.5-flash-lite-20260824/` holds three scenes whose inputs are
paired with request-bound sanitized provider fixtures, in provider-fixture
order:

1. `clear-straight`
2. `stopped-vehicle`
3. `red-light-with-traffic`

Each scenario records prompt, PNG, RGB, dimensions, timestamp, and scenario
metadata hashes, and the corresponding `sanitized_capture` fixtures reject any
different prompt or RGB content.

Replay without network access:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-vertex-fixtures-20260824.yaml \
  --infer \
  --replay-from fixtures/vla/probes/vertex-gemini-3.5-flash-lite-20260824 \
  --output-dir tmp/vertex-fixture-replay-01 \
  --scenario clear-straight \
  --scenario stopped-vehicle \
  --scenario red-light-with-traffic
```

The expected transport result is 3/3 successful replays. The expected semantic
result is 2/3: the recorded stopped-vehicle response proposes `FOLLOW` at full
cruise speed and is intentionally retained for local-safety exercises.

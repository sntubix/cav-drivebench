# Reviewed VLA probe input catalogs

These catalogs contain exact, model-independent prompts and camera PNGs. They
contain no credentials, project IDs, provider response IDs, or cloud
configuration.

## The recorded catalog for the course notebooks

`catalog-20261002-v5/` holds all eight public probe scenes, captured on 2 October
2026 with [`configs/default.yaml`](../../../configs/default.yaml) and DriveBench's original observation. Each
scene directory is ordinary probe output, schema 4, plus `local_scene.json`:
the `LocalScene` measured with the frame, in the form event logs record it.
`partial-occlusion` was recorded again on 29 October 2026, once static trucks
stood on the road rather than 0.7 m above it; only its frame changed.
`metadrive_starter.recorded_probes` loads them and rebuilds each scene's
`ObservationRequest` exactly as `probe --submission` builds it, so the
Assignment 2 notebooks run a team's `observation.py` on these scenes without
MetaDrive. A test checks that the original observation rebuilt this way is
byte-identical to each saved `prompt.txt`.

The frames were rendered on Apple silicon. Rendering is not byte-reproducible:
two captures on the same machine differed in up to 120 pixels per frame, and
GPU vendors differ more. No provider fixture is matched to these frames, so
they serve on any machine. They also replay without MetaDrive, which
sends the saved original observation, for example to score a semantic-failure
fixture on every scene:

```bash
uv run metadrive-starter probe \
  --config configs/demo-vla-semantic-partial-occlusion.yaml \
  --infer \
  --replay-from fixtures/vla/probes/catalog-20261002-v5 \
  --output-dir tmp/semantic-partial-occlusion-replay-01
```

Re-record after any change to MetaDrive, the probe manifest, the output
contract, or scene measurement, into a new directory, then review the frames
before replacing the catalog:

```bash
uv run python -m metadrive_starter.recorded_probes --output-dir tmp/recorded-probes-01
```

The run also writes `summary.json`, which records absolute paths; leave it out
of the catalog.

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

No provider fixture is matched to these frames, and the frames ship, so they
serve on any machine. They replay in id order, through your observation:

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

These inputs are paired with request-bound sanitized provider fixtures.

`vertex-gemini-3.5-flash-lite-20260824/` contains the three inputs used by the
first capped Vertex evidence run, in provider-fixture order:

1. `clear-straight`
2. `stopped-vehicle`
3. `red-light-with-traffic`

Each scenario records prompt, PNG, RGB, dimensions, timestamp, and scenario
metadata hashes. Do not rerender or edit these files: the corresponding
`sanitized_capture` fixtures deliberately reject any different prompt or RGB
content.

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

# Offline provider fixtures

These schema-v1 JSON artifacts replay the typed `ModelProvider` boundary without network access, credentials, or billable requests. Synthetic fixtures use the current `drivebench-vla-assessment-v5` prompt contract; dated live captures retain their original reviewed contract and exact request hashes.

- `synthetic-keep-lane.json` is a valid assessment with provider/model/token metadata.
- `synthetic-malformed.json` reaches the strict assessment decoder and fails there.
- `synthetic-timeout.json` raises the typed timeout category.
- `lane-change-left/` returns one left-change request followed by `KEEP_LANE` for deterministic road-edge rejection from MetaDrive's leftmost spawn lane.
- `lane-change-right/` returns one right-change request followed by `KEEP_LANE` for deterministic completion and dynamic-abort scenarios.
- `semantic-failures/` holds `stopped-vehicle-keep-lane.json` and `partial-occlusion-missed-vehicle.json`, hand-authored, realistic wrong answers that decode cleanly. They fail in opposite directions, so arbitration needs two techniques to catch both:
  - **Stopped vehicle:** the answer reports the stopped car ahead correctly, then proposes `KEEP_LANE` at cruise speed. That is the failure two unrelated model classes produced on that scene. Hazard agreement passes it; only checking the action and speed against the hazards catches it.
  - **Partial occlusion:** the answer reports the truck as low risk and misses the stopped car behind it, so hazard agreement against the measured scene fails it. The open-loop probe cannot see this: reporting the truck satisfies its "vehicle in front" check.

  Each replays through `configs/demo-vla-semantic-<scene>.yaml`, closed-loop or on its probe scene. Without arbitration, the vehicle avoids a crash on emergency braking alone.
- `vertex-gemini-3.5-flash-lite-20260824/` contains three reviewed live Vertex responses: clear road, stopped lead vehicle, and red light with traffic. Their semantic result is 2/3; the stopped-vehicle response deliberately preserves the unsafe full-cruise `FOLLOW` proposal as safety-layer exercise data.

Every document contains a SHA-256 over its canonical content, excluding only `artifact_sha256`. Loading rejects changed content, duplicate/unknown fields, non-standard numbers, oversized files, invalid typed values, and credential-like text. Fixture writers use exclusive creation and never overwrite an earlier artifact.

`source: synthetic` may omit request hashes so one artificial response can support general tests. `source: sanitized_capture` includes both the exact prompt and RGB SHA-256. It contains no credentials, endpoint headers, project IDs, or unreviewed user data. The checked Vertex fixtures omit provider response IDs while retaining model/version, latency, tokens, and raw model output.

The live fixtures are intentionally request-bound. Their exact reviewed prompt/PNG capture catalog is checked in under `fixtures/vla/probes/vertex-gemini-3.5-flash-lite-20260824`; newly rendered images are not interchangeable evidence. Response replay reports recorded latency but does not sleep.

They ship to teams because they replay only from that catalog: nothing is rendered, so the frames' GPU-dependent bytes never matter. A capture without its exact inputs stays instructor-only, since a frame rendered on another GPU vendor differs by a few pixels and no longer matches.

Run the reusable offline closed-loop demonstration from the repository root:

```bash
uv run metadrive-starter run \
  --config configs/demo-vla-fixture.yaml \
  --submission submission \
  --headless
```

This is transport/runtime regression evidence, not evidence of live model quality.

# VLA pipeline and authority

[← Getting started](getting-started.md) · [DriveBench README](../README.md) · [Next: Model providers →](model-providers.md)

Run all commands from the repository root unless stated otherwise.

## Authority model

The VLM receives an observation and returns an assessment. It never actuates the
vehicle. Local code derives a command from that assessment, validates it against
the current scene, and grants it time-bounded authority. Local planning, PID
control, and the instructor-owned safety floor remain outside model control.

The important chain is:

```text
camera + bounded scene context
  → VLM assessment
  → arbitration
  → locally stamped command
  → command validation
  → local planner and PID controller
  → emergency-safety override
  → MetaDrive action
```

Every student-owned layer obeys the monotone restriction: it may reduce the
authority a proposal receives, but never grant more authority than the local
foundation permits.

## VLA assessment and command validation

The model returns a non-authoritative `VLAAssessment` using the restricted
`HighLevelAction` vocabulary. The output contract offers `KEEP_LANE`, `FOLLOW`,
`SLOW_DOWN`, `STOP`, `YIELD`, `CHANGE_LANE_LEFT`, `CHANGE_LANE_RIGHT`, and
`REQUEST_FALLBACK`. Lane changes execute only with `planner.lane_change_enabled`,
and fall back otherwise. `OVERTAKE` and `PULL_OVER` are no longer offered,
because no planner executes them. The decoder still accepts them, so older
artifacts replay, and a command carrying one falls back.

```json
{
  "scene_summary": "A slower vehicle is ahead in the current lane.",
  "relevant_hazards": [
    {"type": "vehicle", "relative_location": "front", "risk": "medium"}
  ],
  "meta_action": "CHANGE_LANE_LEFT",
  "target_speed_mps": 8.3333333333,
  "confidence": 0.86,
  "brief_justification": "The adjacent lane appears clear."
}
```

`VLAAssessment.to_command(...)` copies only the proposed action, speed,
confidence, and justification. The runtime—not the model—adds the command ID,
source-frame simulation time, and configured action horizon.

`VLACommandValidator` returns both the requested and effective command, a
disposition, and structured reasons:

- `accepted`: the command is unchanged;
- `modified`: local rules changed it, such as forcing `STOP` to zero speed;
- `rejected`: an unsafe lateral action became `KEEP_LANE`;
- `fallback`: malformed, stale, future-dated, expired, low-confidence,
  explicitly-fallback, or invalid-scene input requested the conventional path.

Callers must consume `effective_command`, never the unvalidated request. The
continuous emergency supervisor remains final authority after validation,
planning, and control.

<details>
<summary>Validation configuration</summary>

```yaml
command_validation:
  minimum_confidence: 0.5
  maximum_command_age_s: 2.0
  maximum_clock_skew_s: 0.25
  scene_stale_after_s: 0.2
  maximum_target_speed_mps: 13.88888888888889
  slow_down_speed_mps: 5.555555555555555
  yield_speed_mps: 2.7777777777777777
  lane_change_front_gap_m: 12.0
  lane_change_rear_gap_m: 10.0
  lane_change_minimum_ttc_s: 3.0
```

These are the safety floor's thresholds. The floor is instructor-owned, and no
submission replaces the validator: teams decide how far to
trust an assessment in arbitration, above it. Under `--submission`, `agent.yaml`
may tighten any of these values, never loosen them, and the arbiter receives the
same values as its settings. Cloud and local HTTP payloads pass through the
shared assessment parser before the floor validates them.

</details>

## Arbitration

An `AssessmentArbiter` decides how far to trust each assessment before a
command is derived from it. `VLACommandRuntime` calls `arbitrate(request)` on
the control loop when it collects a decoded assessment. The
`ArbitrationRequest` carries the model's assessment, hazards included, and the
local scene measured at that moment. It also carries the current time, when
the model was asked, and the cruise speed. The command validator then receives
`answer.to_command(...)`, stamped with the pipeline's command ID, issue time,
and horizon, so `VLACommand` and everything downstream are unchanged.

Arbitration obeys the monotone restriction, checked by `monotone_violations`
without reference to the scene:

```text
answer.proposed_target_speed_mps <= assessment.proposed_target_speed_mps
answer.confidence                <= assessment.confidence
answer.proposed_action == assessment.proposed_action, or REQUEST_FALLBACK
```

Only those three fields reach the command; the rest stays as the model reported
it. Declining with `REQUEST_FALLBACK` makes the validator fall back, which also
ends any earlier command's authority. If the arbiter raises, returns something
other than a `VLAAssessment`, or answers outside those bounds, the runtime
declines the assessment itself and reports why.

Between assessments, `review(request)` runs every control tick. The
`ReviewRequest` names the assessment behind the command that holds authority,
or none, and the `ModelFailureCategory` of a request that failed on this tick,
if one did. Returning False ends that command; a review may never extend one.
A review that raises or returns anything but True or False ends it too.
Arbiters are stateful: `begin_episode()` runs with every episode reset. Each
call must return within one control step.

Runs under `--submission` arbitrate with the team's `arbitration.py`, built
from the `command_validation` settings. Everything else uses `DefaultArbiter`,
which endorses every assessment. The run summary counts outcomes in
`vla_metrics.arbitrations_endorsed`, `arbitrations_restricted`, and
`arbitrations_declined`, commands a review ended in `arbitration_revocations`,
and answers that could not stand in `arbitration_problems`. Each
`command_validation` event logs the model's `assessment` beside the
`arbitration` decision, and its `requested_command` is the command the
validator received. An `arbitration_review` event records each review that
ended a command or could not stand.

## Camera and model-input boundary

Camera capture is opt-in so PID and manual demonstrations do not pay its rendering
cost:

```yaml
camera:
  enabled: true
  width: 512
  height: 288
```

`MetaDriveCameraAdapter` captures one unnormalised uint8 image, converts
MetaDrive's CPU-side BGR channel order, and returns an immutable `RGBFrame` of
packed row-major RGB bytes. The same boundary works for onscreen rendering and
headless camera capture.

The deterministic prompt declares the assessment vocabulary and schema. The
image is primary evidence; a bounded, distance-sorted `LocalScene` may provide
supplemental safety facts. Behavioural instructions come from
`vla.prompt_policy` in YAML, while code owns the schema, units, context bounds,
and response limits.

The image and prompt together are the observation, and an `ObservationBuilder`
assembles it from an `ObservationRequest`: the captured frame, the scene, the
speeds, and the configured values. Runs under `--submission` build it with the
team's `observation.py`; everything else, runs without `--submission`
included, uses `DefaultObservationBuilder`, which is `build_vla_prompt` on the
captured frame. The pipeline refuses only a malformed observation, such as a
frame whose capture timestamp changed; frame age is always judged on the
captured frame. An observation whose prompt lacks a line of the output contract
is sent, but the pipeline warns the first time and the run summary counts every
one in `vla_metrics.observations_without_output_contract`.

Assessment decoding accepts one JSON object, one JSON code fence, or an exact
`{"assessment": {...}}` wrapper. It rejects prose, multiple or ambiguous
objects, duplicate/unknown fields, non-standard numbers, malformed JSON, and
schema-invalid assessments. The pipeline also checks request correlation and
camera freshness before adding locally owned command metadata.

## Provider and scheduler boundary

`ModelProvider.generate(...)` is synchronous, but `VLAInferenceScheduler` runs
it on one background worker so inference never blocks the control loop.
`ModelRequest` carries the prompt, camera frame, and prompt-contract version.
`ModelResponse` carries raw text, model identity, correlation, latency, and
optional provider/version/token metadata.

```python
from metadrive_starter.vla import InferenceSuccess
from metadrive_starter.vla_runtime import build_vla_scheduler

with build_vla_scheduler(config.vla) as scheduler:
    scheduler.submit(
        frame,
        now_s=simulation_time,
        request_id="run-1-command-42",
        scene=local_scene,
        ego_speed_mps=current_speed_mps,
        cruise_speed_mps=config.controller.target_speed_mps,
    )

    completion = scheduler.poll()
    if isinstance(completion, InferenceSuccess):
        decision = safety_policy.validate(
            completion.result.requested_command,
            local_scene,
            now_s=simulation_time,
        )
```

The scheduler allows one in-flight request and never queues stale frames.
Submission reports `started`, `busy`, `rate_limited`, `request_cap_exhausted`,
or `closed`; polling returns one typed success or failure.

Every request carries an episode ID and monotonically increasing generation ID.
Starting a new episode clears model authority and changes the scheduler epoch.
An older worker may finish, but identity checks turn it into an auditable
`inference_discarded` event rather than executable authority.

## Request budgets and pacing

`vla.maximum_requests_per_run` bounds provider attempts across the complete run.
The budget is consumed immediately before dispatch, so provider success, error,
and timeout all count; camera failures, rate-limited frames, and pre-provider
faults do not. Episode changes do not reset it.

When the cap is exhausted, the runtime stops capturing new model frames, emits
one exhaustion event, lets existing authority expire, and continues with local
fallback and safety. Run summaries record attempted and remaining requests.

Real-time pacing is independent. Use `simulator.realtime: true` and a suitable
`realtime_factor` whenever model latency matters. Pacing never changes simulation
timestamps, command age, action horizon, PID time step, or fault timing.

## Local scene and continuous safety

The safety scene may come from the deterministic oracle or MetaDrive's LiDAR
detections. `DualSceneObserver` can run both against the same state and compare
matched, missed, and extra objects; distance, closing-speed, and TTC errors;
scene validity; and in-path classification.

Curved-path prediction lives behind `FuturePathProjector`. The supplied
`PolylineFuturePath` projects objects onto sampled future route segments and
reports along-path distance, cross-track offset, and path-relative velocity.
The perception and emergency-safety layers consume that contract rather than its
geometry implementation.

The emergency supervisor continuously evaluates closing threats after ordinary
control. It can suppress throttle or apply service/emergency braking, but does
not steer around obstacles.

## Action-speed and headway policies

The model proposes a raw target speed, but local action semantics can map that
proposal to a course-owned speed. `vla.action_speed_policy_mode` supports:

- `enforce`: apply the mapped speed;
- `shadow`: record what would have changed while preserving the model speed;
- `off`: preserve the model speed and report no intervention.

The independent headway-speed cap has the same `enforce`, `shadow`, and `off`
modes. It computes a route-aware speed ceiling from lead gap, lead speed,
minimum gap/headway, reaction time, and maximum deceleration. Emergency braking
remains final authority in every mode.

## Lane-change authority

Lateral actions fail closed unless the adjacent lane exists and passes configured
front/rear gap and TTC checks. After admission, the lane-change coordinator
continues checking clearance, route generation, progress, timeout, and completion.
It may abort an admitted manoeuvre when conditions change. A model request never
bypasses continuous local geometry checks.

## Where to go next

Continue with [Model providers](model-providers.md) to run deterministic fixtures,
supported local servers, or the instructor-funded Vertex path.

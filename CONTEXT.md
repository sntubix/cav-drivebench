# DriveBench

DriveBench is an assessed course project in which student teams build the decision
system around an unreliable vision-language model driving a simulated vehicle. This
glossary fixes the vocabulary that the codebase, the assignment briefs, and the
grading material must all share.

## Language

### The model and its output

**VLM**:
A vision-language model that receives an observation and returns a structured
assessment. It never actuates the vehicle.
_Avoid_: VLA model, driving model, the agent

**VLA subsystem**:
The pipeline in which a VLM advisor proposes an assessment and local code derives,
validates, and executes a command. The `vla_` prefix throughout the codebase names
this subsystem, not the model.
_Avoid_: the VLA, the VLA model

**Assessment**:
One validated VLM output describing a scene, its hazards, a proposed manoeuvre, a
proposed target speed, and a confidence. Descriptive; carries no authority.
_Avoid_: prediction, decision, action, response

**Command**:
A target speed and manoeuvre derived locally from an assessment and stamped with
locally owned execution metadata. Only a command can reach planning.
_Avoid_: model output, model action

**Authority**:
Permission for a command to influence the vehicle. Authority is granted by local
code, is bounded in time, and is never held by the model.
_Avoid_: control, ownership

### The layers

**Observation**:
Everything the model receives for one request: the rendered image together with the
textual context and prompt that accompany it. One artifact, assembled by student code
from values declared in configuration.
_Avoid_: input, prompt, frame

**Output contract**:
The prompt lines that ask for exactly what the decoder accepts: the contract version, the
required output fields, the allowed actions, and the JSON-only instruction. Supplied to
every observation as a known-good default that teams may move, rephrase, or replace; a
run reports every observation that lacks one of its lines.
_Avoid_: schema prompt, contract block

**Request policy**:
The rule deciding when to ask the model, under a bounded request budget.
_Avoid_: scheduler, cadence, rate limit

**Arbitration**:
The decision about how far to trust an assessment, made across successive assessments
and against the local scene. Arbitration may only reduce the authority a model
proposal receives.
_Avoid_: filtering, validation, post-processing

**Hazard agreement**:
Whether the hazards in an assessment match the local scene it is validated against. An
assessment agrees when it reports no hazard the scene lacks and leaves out none in the
ego path. A signal arbitration can compute; it cannot catch a correctly reported hazard
paired with an unsafe manoeuvre.
_Avoid_: hazard accuracy, perception score

**Safety floor**:
The instructor-owned checks that no submission can weaken: physical clamps, scene
validity, geometric clearance, and emergency braking. Distinct from arbitration,
which sits above it and may be stricter.
_Avoid_: safety layer, validator, supervisor

**Monotone restriction**:
The invariant that every student-owned layer may only remove authority, never grant
it. A broken submission can therefore be over-conservative but never unsafe.

### Evaluation

**Probe**:
A fixed open-loop scenario in which one assessment is scored against an expected
manoeuvre and target-speed range. Measures semantic quality without driving.
_Avoid_: test, benchmark, eval

**Public scenario**:
A scenario teams may run freely for their own reporting. Never contributes to a grade.
_Avoid_: practice test, sample

**Hidden scenario**:
A withheld scenario run only on instructor infrastructure. The sole source of graded
performance numbers.
_Avoid_: private test, final test

**Structural gate**:
A mechanical pass/fail check on a submission — budget respected, control loop never
blocked, no unhandled exception, monotone restriction preserved. Independent of model
quality and of model nondeterminism.
_Avoid_: smoke test, sanity check

**Score**:
The quality measure out of 100 that a gate run reports beside its structural gates, from
implementation checks and from how the submission drove the scenarios. Never decides a gate. On
the hidden scenarios, it is Assignment 1's mark.
_Avoid_: grade, mark

**Implementation check**:
A deterministic check of a submitted controller on its own, driven against a toy vehicle
outside the simulator with gains the check chooses. Tests whether a control term works,
never a team's tuning.
_Avoid_: bench check, unit test, probe

**Par**:
The steps the reference controller takes to arrive on one scenario: the yardstick for a
submission's time.
_Avoid_: baseline, expected time

**Fault**:
A deliberately injected failure of the camera, LiDAR, or model provider, scheduled at
a fixed simulation time. The instrument that makes arbitration gradeable
deterministically.
_Avoid_: error, failure case

**Sound assessment**:
An assessment whose proposed manoeuvre a probe accepts and whose proposed target speed
lies within that probe's range; any other is unsound. Judged against what the probe
expects, which arbitration never sees.
_Avoid_: correct assessment, safe assessment, good answer

**Catch rate**:
The share of unsound assessments from which arbitration takes authority: declined, held
below the confidence threshold, or restricted into sound ones.
_Avoid_: recall, detection rate

**Keep rate**:
The share of sound assessments that arbitration leaves sound and above the confidence
threshold.
_Avoid_: precision, pass rate

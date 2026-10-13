# DriveBench

DriveBench is an assessed course project in which student teams build the
decision system around an unreliable vision-language model (VLM) driving a
simulated vehicle. MetaDrive simulates the world and the vehicle; DriveBench
supplies the local control, safety, model-provider, evaluation, and replay
boundaries around it.

## Start here

1. [Get the code and install it](docs/getting-started.md).
2. Read the brief of the current assignment:
   [Assignment 1: PID control](course/assignment-1/README.md).
3. Write your work in `submission/`, [check it](docs/getting-started.md#check-your-work)
   with `uv run metadrive-starter gates`, and
   [hand it in](docs/getting-started.md#hand-in) on Moodle.

## How the project works

Your team writes code only in `submission/`, never in files already in the
tree. Each assignment adds one piece of the decision system around the model,
and every file ships as a working stub, so the whole pipeline drives from the
first day:

| Assignment | You write | It decides |
| --- | --- | --- |
| 1. PID control | `controller.py` | steering, throttle, and brake from tracking errors |
| 2. Observation and arbitration | `observation.py`, `arbitration.py` | what the model sees, and how far to trust its assessment |
| 3. Request cadence and cloud | `request_policy.py` | when to ask the model, within a request budget |
| 4. Race | no new code | tuning only |

`submission/agent.yaml` holds the tuning values your assignment permits.
Everything else, including the simulator, route tracking, the safety floor,
model providers, and evaluation, is instructor-owned foundation that your code
calls but never edits. Run the public routes as often as you like; graded
numbers come only from hidden routes that the instructors run.

All assignments in the published course sequence are required. Instructors
provide the cloud accounts used by the course, and the supplied profiles enforce
request caps. Students do not provision cloud projects or administer billing.

## Reference

- [Running and configuring DriveBench](docs/running-and-configuring.md):
  rendered demonstrations, pacing, platforms, Docker, and configuration.
- [Glossary](CONTEXT.md): the course's vocabulary.
- From Assignment 2 on: [VLA pipeline and authority](docs/vla-pipeline.md),
  [Model providers](docs/model-providers.md), and
  [Evaluation and replay](docs/evaluation.md).

<details>
<summary>Repository layout</summary>

```text
configs/                         simulator, controller, provider, and evaluation profiles
src/metadrive_starter/           application package and CLI
src/metadrive_starter/controllers/ vehicle-controller protocol and factory
src/metadrive_starter/planning/  route tracking and future-path projection
src/metadrive_starter/perception/ local-scene adapters
src/metadrive_starter/safety/    command validation and emergency safety
src/metadrive_starter/vla/       camera, transport, assessment, and scheduling
scripts/platform-check.sh        Linux Docker validation and evidence capture
scripts/native-platform-check.sh native Ubuntu VLM validation and evidence capture
tests/                           unit, integration, and simulator checks
submission/                      team-owned code, starting from working stubs
course/assignment-1/             Assignment 1 brief and PID lab notebook
```

</details>

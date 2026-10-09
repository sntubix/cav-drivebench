# Model providers

[← VLA pipeline and authority](vla-pipeline.md) · [DriveBench README](../README.md) · [Next: Evaluation and replay →](evaluation.md)

Run all commands from the repository root unless stated otherwise.

## Supported provider paths

All providers feed the same assessment decoder, command derivation, validation,
scheduler, and controller. The provider changes where inference occurs, not the
vehicle-authority boundary.

Only officially recommended engines are eligible for course support. Exact
runtime/model profiles become supported after they pass their platform
acceptance sequence:

| Platform | Official path | Current status |
| --- | --- | --- |
| Any | checked fixtures | supported deterministic development path |
| Apple Silicon | MLX-VLM | supported local VLM server |
| Linux | `llama.cpp`/`llama-server` | supported engine; pinned native candidate and acceptance validator available |
| Course cloud | Vertex | supported instructor-funded coursework path |

Other OpenAI-compatible servers may work, but support begins only after an exact,
pinned runtime/model combination passes the probe, closed-loop, replay, and
resource acceptance sequence.

## Offline fixture provider

The fixture provider loads strict, versioned JSON outcomes and performs no
network I/O. It is the first model-facing step because it exercises the complete
runtime and safety chain deterministically:

```bash
uv run metadrive-starter run \
  --config configs/demo-vla-fixture.yaml \
  --submission submission \
  --headless
```

The checked fixtures cover synthetic success, malformed-response, timeout, and
lane-change outcomes. Reviewed request-bound fixtures retain exact prompt and RGB
hashes. Loading rejects tampering, duplicate/unknown fields, non-standard values,
credential-like text, and incompatible prompt contracts. Student releases include
the fixtures from Assignment 2 on.

## OpenAI-compatible HTTP provider

The HTTP provider sends an OpenAI-compatible multimodal chat-completions request.
It includes the prompt and lossless PNG data URL, then accepts the returned
`choices[0].message.content` for strict local decoding.

The checked generic probe profile targets a loopback server on port 8000:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-local.yaml \
  --infer \
  --output-dir tmp/vla-probe-model-01
```

Existing output directories are never overwritten. Use a new directory for
every capture or provider run.

<details>
<summary>HTTP provider configuration</summary>

```yaml
camera:
  enabled: true

vla:
  enabled: true
  provider: http
  maximum_requests_per_run: null
  request_timeout_s: 10.0
  minimum_interval_s: 0.5
  action_horizon_s: 2.0
  max_scene_objects: 8
  maximum_frame_age_s: 0.5
  maximum_clock_skew_s: 0.05
  http:
    endpoint_url: http://127.0.0.1:8000/v1/chat/completions
    model_id: local-vlm
    api_key_env: null
    max_response_bytes: 1048576
    max_tokens: 256
    allow_insecure_http: false
    structured_output: false
    temperature: 0.0
    seed: null
```

Loopback HTTP is accepted by default. Non-loopback endpoints require HTTPS unless
`allow_insecure_http` is deliberately enabled for a trusted host-only network.
Redirects are not followed, preventing authorization headers from being forwarded
to another endpoint. `api_key_env` names an environment variable; secrets are
redacted from diagnostics.

Structured output is an explicit test dimension. A supporting server receives a
strict JSON Schema when `structured_output: true`; the local decoder remains
authoritative in either mode.

`temperature` is sent with every request; 0 asks for the most likely answer each
time. The `llama-server` profiles for SmolVLM-256M and the pinned Qwen3-VL-2B
sample at 0.7, as Assignment 2's grading does, and so do the MLX profiles.
They set no `seed`, so each request samples under the server's default:
`llama-server --seed`, or 0 under `mlx_vlm.server`, which has no such flag.
Under one seed the same input gets the same answer, so pass `--seed n` to
`probe` or `run`, which sends seed n with every request, rerun under a new one
each time, and report a live figure as the median and range over those runs.
`seed`, when set, fixes the server's sampling so that a sampled run can be
repeated exactly. Both are instructor-owned
under `vla.http`, so `agent.yaml` cannot set them.

</details>

## Linux local-server acceptance: llama.cpp

The official Linux engine is `llama.cpp`'s OpenAI-compatible multimodal
`llama-server`. The native CPU candidate is deliberately immutable:

- llama.cpp `b10771`, official Ubuntu x64 archive SHA-256
  `42bb60d6027c99ec05ff1a5ae441e45345a3fcccaf86ca655b4dc51b5519c814`;
- Qwen model revision `d38d39f5972e27cd58023f9b1e9f994b0c85ca47`;
- `Qwen3VL-2B-Instruct-Q4_K_M.gguf`, SHA-256
  `089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae`;
- `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`, SHA-256
  `f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82`.

The archive digest comes from the
[llama.cpp build attestation](https://github.com/ggml-org/llama.cpp/attestations/44860492),
and the model files come from the immutable
[Qwen GGUF revision](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/d38d39f5972e27cd58023f9b1e9f994b0c85ca47).

On a native Ubuntu x86_64 checkout, install `curl`, `Xvfb`, and GNU `time`, then
download the three artifacts into the ignored `tmp/` area:

```bash
mkdir -p tmp/native-vlm/llama-b10771

curl --fail --location \
  --output tmp/native-vlm/llama-b10771-bin-ubuntu-x64.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b10771/llama-b10771-bin-ubuntu-x64.tar.gz
printf '%s  %s\n' \
  42bb60d6027c99ec05ff1a5ae441e45345a3fcccaf86ca655b4dc51b5519c814 \
  tmp/native-vlm/llama-b10771-bin-ubuntu-x64.tar.gz | sha256sum --check
tar -xzf tmp/native-vlm/llama-b10771-bin-ubuntu-x64.tar.gz \
  -C tmp/native-vlm/llama-b10771

curl --fail --location \
  --output tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/d38d39f5972e27cd58023f9b1e9f994b0c85ca47/Qwen3VL-2B-Instruct-Q4_K_M.gguf
curl --fail --location \
  --output tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/d38d39f5972e27cd58023f9b1e9f994b0c85ca47/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
printf '%s  %s\n%s  %s\n' \
  089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae \
  tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82 \
  tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf | sha256sum --check
```

Find the extracted server path, make sure the Git checkout is clean, and run the
single acceptance command:

```bash
LLAMA_SERVER=$(find tmp/native-vlm/llama-b10771 \
  -type f -name llama-server -print -quit)

./scripts/native-platform-check.sh \
  --llama-archive tmp/native-vlm/llama-b10771-bin-ubuntu-x64.tar.gz \
  --llama-server "${LLAMA_SERVER}" \
  --model tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
```

The validator refuses WSL, containers, non-Ubuntu/non-x86_64 hosts, dirty
checkouts, version drift, and hash drift. It creates the frozen native `uv`
environment, runs the native simulator baseline, captures one probe corpus, and
reuses those exact inputs for schema-off and schema-on inference. Schema-off
decoding is evidence only, recorded in each round's `acceptance.json`; every
schema-on scene must decode. It then runs a
paced, request-capped closed loop that must receive successful responses, grant
model authority, avoid runtime errors/collision/off-road termination, and replay
with zero mismatches. The complete model sequence is repeated after a clean
server restart. Timing, peak simulator RSS, sampled server RSS, commands, logs,
host identity, and artifact hashes are written under
`artifacts/native-platform-validation/<UTC timestamp>/` without overwriting an
earlier run.

Probe semantic scores remain evidence for model-quality work; they do not turn a
working transport/runtime qualification into a failure. CPU-only settings make
the accepted baseline independent of a discrete GPU. Accelerated variants need
their own complete evidence bundle before they become supported profiles. The
server and model remain outside the DriveBench `uv` environment.

### SmolVLM on the same server: the Assignment 2 experiment

Whether teams can develop against SmolVLM-256M on Linux decides part of
Assignment 2. The validator answers it as an optional experiment once the
simulator baseline (N0-N4) passes, whatever the Qwen rounds' result, with the
files the 3 September Linux container trial used, from the immutable
[ggml-org GGUF revision](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/tree/b9e4379657e1450d04d02eec8e345667265b0a00):

- `SmolVLM-256M-Instruct-Q8_0.gguf`, SHA-256
  `2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65`;
- `mmproj-SmolVLM-256M-Instruct-Q8_0.gguf`, SHA-256
  `7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd`.

```bash
curl --fail --location \
  --output tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/b9e4379657e1450d04d02eec8e345667265b0a00/SmolVLM-256M-Instruct-Q8_0.gguf
curl --fail --location \
  --output tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/b9e4379657e1450d04d02eec8e345667265b0a00/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
printf '%s  %s\n%s  %s\n' \
  2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65 \
  tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd \
  tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf | sha256sum --check
```

Add `--smolvlm-model tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf` and
`--smolvlm-mmproj tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf` to the
acceptance command. Gate `n7-smolvlm` then serves SmolVLM under JSON Schema
output ([`configs/vla-probe-smolvlm-llama-structured.yaml`](../configs/vla-probe-smolvlm-llama-structured.yaml)), requires every
probe scene to decode, runs a team's closed loop with `--submission submission`
([`configs/demo-vla-smolvlm-llama.yaml`](../configs/demo-vla-smolvlm-llama.yaml)) with zero runtime errors, and replays
it with zero mismatches. Model authority is recorded but not required: the
safety floor is expected to reject SmolVLM's answers for low confidence. The
result never changes the validator's overall result.

## Apple Silicon local server: MLX-VLM

Keep the MLX runtime outside the project environment. Start the supported Qwen
checkpoint on port 8080:

```bash
uvx --from mlx-vlm==0.6.8 --with torchvision mlx_vlm.server \
  --model mlx-community/Qwen3-VL-2B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8080
```

Then run the matching open-loop probe:

```bash
uv run metadrive-starter probe \
  --config configs/vla-probe-qwen3-vl-2b-mlx.yaml \
  --infer \
  --output-dir tmp/vla-probe-qwen3-vl-2b-01
```

After the probe succeeds, run the paced closed loop:

```bash
uv run metadrive-starter run \
  --config configs/demo-vla-qwen3-vl-2b-mlx.yaml \
  --submission submission
```

<details>
<summary>Weak local checkpoint for early experiments</summary>

SmolVLM is the low-resource teaching profile. It is useful for checking model
loading, image transport, prompt iteration, and raw-response inspection; it is
not expected to be a viable driving policy.

```bash
uvx --from mlx-vlm==0.6.8 --with torchvision mlx_vlm.server \
  --model mlx-community/SmolVLM-256M-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8080

uv run metadrive-starter probe \
  --config configs/vla-probe-smolvlm-mlx.yaml \
  --infer \
  --output-dir tmp/vla-probe-smolvlm-01
```

</details>

MLX now offers Linux backends, so an equivalent isolated `uvx` experiment may be
evaluated on suitable Linux hardware. It is not a supported course profile until
that exact engine/backend/model combination passes the Linux acceptance sequence;
the llama.cpp path remains the baseline.

## Vertex course provider

Vertex is part of the required coursework. Instructors provide the account and
course project to use, and apply provider-side cost controls. The supplied
DriveBench profiles add a per-run request cap; students must not remove or raise
that cap.

Install the optional cloud dependencies:

```bash
uv sync --group dev --extra cloud
```

Create an ignored `.env` using the project identifier supplied by the instructors:

```dotenv
GOOGLE_CLOUD_PROJECT=the-instructor-provided-project
```

Load it and authenticate with the provided account:

```bash
set -a
source .env
set +a
gcloud auth application-default login
```

Check credentials, project resolution, text/image requests, strict schema, and
latency before a driving run:

```bash
uv run metadrive-starter cloud-check --samples 3
```

This exact check makes four potentially billable requests: one text request and
three image/schema samples. It is not governed by
`vla.maximum_requests_per_run`. Run it only after the instructors confirm that
the account's enforceable provider-side quota or spend cap is active. The YAML
request cap below applies to the subsequent DriveBench run, not to preflight.

Run the request-capped course profile:

```bash
uv run metadrive-starter run \
  --config configs/demo-vla-vertex.yaml \
  --submission submission
```

[`configs/demo-vla-vertex.yaml`](../configs/demo-vla-vertex.yaml) permits at most ten provider attempts and applies
half-speed simulation pacing. The cap counts attempts before dispatch, so errors
and timeouts still consume budget. Local command validation, headway enforcement,
and emergency safety remain active regardless of provider output.

Instructor provisioning, quotas, billing administration, and hidden evaluation
operations live only in the instructor documentation.

<details>
<summary>Reviewed cloud capture and offline conversion</summary>

The fixed cloud-capture workflow performs exactly five attempts: two preflight
requests followed by three exact open-loop scenes. It requires explicit
`--execute` confirmation and never retries.

```bash
uv run metadrive-starter probe \
  --scenario clear-straight \
  --scenario stopped-vehicle \
  --scenario red-light-with-traffic \
  --output-dir tmp/vertex-inputs-01

uv run metadrive-starter vertex-capture \
  --replay-from tmp/vertex-inputs-01 \
  --output-dir tmp/vertex-capture-01 \
  --execute
```

After policy review, convert captured responses into sanitized offline fixtures:

```bash
uv run metadrive-starter provider-fixtures \
  --source-dir tmp/vertex-capture-01/probes \
  --output-dir tmp/reviewed-provider-fixtures \
  --fixture-prefix vertex-course \
  --scenario clear-straight \
  --scenario stopped-vehicle \
  --scenario red-light-with-traffic \
  --reviewed
```

Provider response IDs remain excluded unless staff explicitly retain them after
policy review. Credentials, project IDs, private data, and unreviewed/raw
captures must not enter student submissions or public artifacts. Reviewed,
sanitized prompt/image fixtures may be published through the checked fixture
workflow.

</details>

## In-process model option

An experiment may inject a `ModelProvider` factory and keep generation in the
simulator process:

```python
def provider_factory(settings):
    return ResearchModelProvider(
        model_id=settings.http.model_id,
        max_tokens=settings.http.max_tokens,
    )

with build_vla_scheduler(
    config.vla,
    http_provider_factory=provider_factory,
) as scheduler:
    ...
```

The factory receives validated settings and must return an object implementing
`generate(ModelRequest) -> ModelResponse`. YAML never imports arbitrary classes.
In-process execution is an experiment boundary, not the default course runtime:
it shares failures and memory pressure with the simulator and cannot be forcibly
cancelled safely.

Continue with [Evaluation and replay](evaluation.md) after the selected provider
passes the fixed open-loop probe.

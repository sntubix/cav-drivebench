# Model providers

[← VLA pipeline and authority](vla-pipeline.md) · [DriveBench README](../README.md) · [Next: Evaluation and replay →](evaluation.md)

You need this guide from Assignment 2 on; Assignment 1 uses none of it.

Run all commands from the repository root unless stated otherwise.

## Supported provider paths

All providers feed the same assessment decoder, command derivation, validation,
scheduler, and controller. The provider changes where inference occurs, not the
vehicle-authority boundary.

| Platform | Path | Use |
| --- | --- | --- |
| Any | checked fixtures | deterministic development, and the gates |
| Linux | `llama.cpp`'s `llama-server`, the pinned build | Assignment 2's marked work; grading runs its Vulkan build |
| Apple Silicon | the same pinned `llama-server`, on Metal | Assignment 2's marked work |
| Apple Silicon | MLX-VLM | demos and plumbing checks only |
| Course cloud | Vertex | instructor-funded coursework path |

Other OpenAI-compatible servers may work with the HTTP provider, but the course
supports only these.

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
authoritative in either mode. `llama-server` `b10771` holds an answer to the
schema's shape but not to its numeric bounds: under it, SmolVLM still returned
confidences of 1.006 and -1.007 and a target speed of -0.026 m/s, on macOS and
Linux, on CPU and GPU. The decoder rejects such an answer, so it counts as one
that does not decode.

`temperature` is sent with every request; 0 asks for the most likely answer each
time. The `llama-server` profiles for SmolVLM-256M and the pinned Qwen3-VL-2B
sample at 0.7, as Assignment 2's grading does, and so do the MLX profiles.
They set no `seed`, so each request samples under the server's default:
`llama-server --seed`, or 0 under `mlx_vlm.server`, which has no such flag.
Under one seed the same input gets the same answer, so pass `--seed n` to
`probe` or `run`, which sends seed n with every request, rerun under a new one
each time, and report a live figure as the median and range over those runs.
`seed`, when set, fixes the server's sampling so that a sampled run can be
repeated exactly on the same machine, provided the server reads each prompt in
full: `llama-server` reuses the prompts it has cached, and an answer that
reuses one can differ from the first answer to the same request and seed.
Serve with `--no-cache-prompt` when runs must repeat exactly, as Assignment 2's
grading does. Both are instructor-owned
under `vla.http`, so `agent.yaml` cannot set them.

</details>

## The pinned `llama-server` and model files

Assignment 2's marked work, and its grading, run `llama.cpp`'s OpenAI-compatible
multimodal `llama-server` at one pinned build, `b10771`, serving pinned model
files. Check every file by SHA-256 before use: on Linux with `sha256sum --check`,
as below, and on macOS with `shasum -a 256 -c`, which takes the same input.

- llama.cpp `b10771`, Ubuntu x64 archive `llama-b10771-bin-ubuntu-x64.tar.gz`,
  SHA-256 `42bb60d6027c99ec05ff1a5ae441e45345a3fcccaf86ca655b4dc51b5519c814`,
  from the
  [llama.cpp build attestation](https://github.com/ggml-org/llama.cpp/attestations/44860492);
  the Vulkan and macOS arm64 archives of the same build are listed below.
- Qwen3-VL-2B-Instruct, from the immutable
  [Qwen GGUF revision](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/d38d39f5972e27cd58023f9b1e9f994b0c85ca47)
  `d38d39f5972e27cd58023f9b1e9f994b0c85ca47`: `Qwen3VL-2B-Instruct-Q4_K_M.gguf`,
  SHA-256 `089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae`, and
  `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`, SHA-256
  `f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82`.
- SmolVLM-256M-Instruct, from the immutable
  [ggml-org GGUF revision](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/tree/b9e4379657e1450d04d02eec8e345667265b0a00)
  `b9e4379657e1450d04d02eec8e345667265b0a00`: `SmolVLM-256M-Instruct-Q8_0.gguf`,
  SHA-256 `2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65`, and
  `mmproj-SmolVLM-256M-Instruct-Q8_0.gguf`, SHA-256
  `7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd`.

The server and the model files stay outside the DriveBench `uv` environment, in
the ignored `tmp/` area. Download the four model files once:

```bash
mkdir -p tmp/native-vlm

curl --fail --location \
  --output tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/d38d39f5972e27cd58023f9b1e9f994b0c85ca47/Qwen3VL-2B-Instruct-Q4_K_M.gguf
curl --fail --location \
  --output tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/d38d39f5972e27cd58023f9b1e9f994b0c85ca47/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
curl --fail --location \
  --output tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/b9e4379657e1450d04d02eec8e345667265b0a00/SmolVLM-256M-Instruct-Q8_0.gguf
curl --fail --location \
  --output tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/b9e4379657e1450d04d02eec8e345667265b0a00/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae \
  tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82 \
  tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
  2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65 \
  tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd \
  tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf | sha256sum --check
```

Then the Linux CPU server, or [the Vulkan build](#gpu-the-vulkan-build-of-the-same-pin)
on a GPU, or [the macOS arm64 build](#apple-silicon-the-macos-arm64-build-of-the-same-pin)
on Apple silicon:

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
```

### Serving a model for Assignment 2

Assignment 2 serves one model at a time on port 8000, with the files
downloaded above. For SmolVLM:

```bash
LLAMA_SERVER=$(find tmp/native-vlm/llama-b10771 -type f -name llama-server -print -quit)
"${LLAMA_SERVER}" -m tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  --mmproj tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --host 127.0.0.1 --port 8000 --alias drivebench-smolvlm-256m-q8-0 \
  --ctx-size 4096 --threads 4 --n-gpu-layers 0 --no-mmproj-offload --parallel 1 \
  --temp 0 --seed 0
```

For Qwen, pass `tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf` and
`--mmproj tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` instead, with
`--alias drivebench-qwen3-vl-2b-q4-k-m` and `--image-min-tokens 1024`. Each
Assignment 2 configuration names the alias it expects.

Serve with `--no-cache-prompt`, as grading does, when a run must repeat
exactly. On Linux with a GPU, serve
[the Vulkan build](#gpu-the-vulkan-build-of-the-same-pin), which grading uses,
far faster than the CPU. On Apple silicon, do all marked work on
[the macOS arm64 build](#apple-silicon-the-macos-arm64-build-of-the-same-pin),
on Metal: [MLX](#apple-silicon-local-server-mlx-vlm) serves other checkpoints,
whose figures do not predict the marks.

SmolVLM runs on any laptop. Qwen3-VL-2B is about ten times slower per answer
and, without a discrete GPU, can near the 60 s request timeout, so run it on
your team's strongest laptop, together, in planned sessions: the server stays
on that machine's loopback. Qwen's server needs about 4 GiB of memory and
SmolVLM's under 1 GiB, beside about 5 GiB for the simulator, whose camera
render slows every closed-loop run.

### GPU: the Vulkan build of the same pin

`b10771` has no Linux CUDA build. Its Vulkan build runs the same pinned model
files on a GPU through the vendor's Vulkan driver, NVIDIA's included:

- `llama-b10771-bin-ubuntu-vulkan-x64.tar.gz`, SHA-256
  `35d98f9c60dcd6c1af2267990b33fcd981a011f8b902dcbb0d5f280bf807bd53`.

Download, check, and unpack it as the CPU archive above, into
`tmp/native-vlm/llama-b10771-vulkan`, and serve with the same flags, putting
`--n-gpu-layers 99` in place of `--n-gpu-layers 0 --no-mmproj-offload`.
Assignment 2 is graded on this build. On an NVIDIA GPU it honours the request's
seed and, with `--no-cache-prompt`, gives the same request and seed the same
answer however often it is asked, as the CPU build does; the answers
themselves differ from the CPU build's, as they differ between machines. The
first request after installing is slow while the driver compiles its shaders.

### Apple silicon: the macOS arm64 build of the same pin

On Apple silicon, do all of Assignment 2's marked work on `b10771`'s macOS arm64
build, on Metal, serving the same pinned model files:

- `llama-b10771-bin-macos-arm64.tar.gz`, SHA-256
  `371bb6c01543c8b748e1404e23a5cb09b2bc7484df9b90febf5c068f04adec33`.

Download the archive, and the four model files with the `curl` commands above,
then check them all with `shasum`:

```bash
mkdir -p tmp/native-vlm/llama-b10771-macos

curl --fail --location \
  --output tmp/native-vlm/llama-b10771-bin-macos-arm64.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b10771/llama-b10771-bin-macos-arm64.tar.gz
printf '%s  %s\n' \
  371bb6c01543c8b748e1404e23a5cb09b2bc7484df9b90febf5c068f04adec33 \
  tmp/native-vlm/llama-b10771-bin-macos-arm64.tar.gz | shasum -a 256 -c
tar -xzf tmp/native-vlm/llama-b10771-bin-macos-arm64.tar.gz \
  -C tmp/native-vlm/llama-b10771-macos

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae \
  tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82 \
  tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
  2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65 \
  tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd \
  tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf | shasum -a 256 -c
```

Serve one model at a time on port 8000, as above, with every layer on Metal and
the prompt cache off. For SmolVLM:

```bash
LLAMA_SERVER=$(find tmp/native-vlm/llama-b10771-macos -type f -name llama-server -print -quit)
"${LLAMA_SERVER}" -m tmp/native-vlm/SmolVLM-256M-Instruct-Q8_0.gguf \
  --mmproj tmp/native-vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --host 127.0.0.1 --port 8000 --alias drivebench-smolvlm-256m-q8-0 \
  --ctx-size 4096 --threads 4 --n-gpu-layers 99 --parallel 1 \
  --no-cache-prompt --temp 0 --seed 0
```

For Qwen, pass `tmp/native-vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf` and
`--mmproj tmp/native-vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` instead, with
`--alias drivebench-qwen3-vl-2b-q4-k-m` and `--image-min-tokens 1024`.
`"${LLAMA_SERVER}" --list-devices` should list a Metal device. The
`vla-probe-*-llama` configurations of the brief run against this server as they
are, so `--seed n` works as on Linux: with `--no-cache-prompt` the same request
and seed get the same answer. A prompt counts as many tokens as on the grading
server, and the measures point the way they do there, though the answers
themselves differ, as they differ between machines. Measure your own latency, as
[Choosing a realtime factor](running-and-configuring.md#choosing-a-realtime-factor)
says.

## Apple Silicon local server: MLX-VLM

MLX serves demos and plumbing checks only, never Assignment 2's marked work. Its
checkpoints are not the pinned GGUF files and answer differently, so observation
figures from MLX do not predict the marks: syntax and scene interpretation can
point the opposite way from the pinned `llama-server`, and SmolVLM has no MLX
profile under JSON Schema output. Use
[the macOS arm64 build](#apple-silicon-the-macos-arm64-build-of-the-same-pin) for
those.

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

## Vertex course provider

Vertex is part of the required coursework. Instructors provide the account and
course project to use, and apply provider-side cost controls. The Vertex
profiles add a per-run request cap, which local-server profiles do not carry;
students must not remove or raise it.

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

## In-process model option

An experiment may build the scheduler around its own `ModelProvider` and keep
generation in the simulator process:

```python
from metadrive_starter.vla import VLAInferencePipeline, VLAInferenceScheduler

provider = ResearchModelProvider(
    model_id=config.vla.http.model_id,
    max_tokens=config.vla.http.max_tokens,
)
with VLAInferenceScheduler(
    VLAInferencePipeline(provider),
    minimum_interval_s=config.vla.minimum_interval_s,
) as scheduler:
    ...
```

The provider must implement `generate(ModelRequest) -> ModelResponse`. YAML
never imports arbitrary classes.
In-process execution is an experiment boundary, not the default course runtime:
it shares failures and memory pressure with the simulator and cannot be forcibly
cancelled safely.

Continue with [Evaluation and replay](evaluation.md) after the selected provider
passes the fixed open-loop probe.

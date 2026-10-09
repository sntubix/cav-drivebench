from __future__ import annotations

import base64
import json
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from typing import Iterator

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.vla import (
    HighLevelAction,
    InferenceSubmitDisposition,
    InferenceSuccess,
    ModelProviderError,
    ModelRequest,
    OpenAICompatibleHTTPProvider,
    RGBFrame,
)
from metadrive_starter.vla_probe import (
    ProbeCapture,
    ProbeScenario,
    run_vla_probe_catalog,
)
from metadrive_starter.vla_runtime import build_http_vla_provider, build_http_vla_scheduler


@contextmanager
def _http_server(
    *,
    response_body: bytes = b"{}",
    redirect_url: str | None = None,
) -> Iterator[tuple[str, dict[str, object]]]:
    observed: dict[str, object] = {"requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            observed["requests"] = int(observed["requests"]) + 1
            observed["body"] = self.rfile.read(content_length)
            observed["authorization"] = self.headers.get("Authorization")
            observed["request_id"] = self.headers.get("X-Request-ID")
            if redirect_url is not None:
                self.send_response(307)
                self.send_header("Location", redirect_url)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1/chat/completions", observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def _completion(scheduler) -> InferenceSuccess:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        completion = scheduler.poll()
        if completion is not None:
            assert isinstance(completion, InferenceSuccess)
            return completion
        Event().wait(0.001)
    raise AssertionError("HTTP inference did not complete")


def test_real_http_transport_runs_through_pipeline_and_scheduler() -> None:
    command = {
        "scene_summary": "Clear road.",
        "relevant_hazards": [],
        "meta_action": "KEEP_LANE",
        "target_speed_mps": 20.0,
        "confidence": 0.9,
        "brief_justification": "No hazards visible.",
    }
    response = json.dumps(
        {"choices": [{"message": {"content": json.dumps(command)}}]}
    ).encode()

    with _http_server(response_body=response) as (endpoint, observed):
        settings = config_from_dict(
            {
                "vla": {
                    "enabled": True,
                    "minimum_interval_s": 0.5,
                    "http": {
                        "endpoint_url": endpoint,
                        "model_id": "loopback-vlm",
                        "api_key_env": "TEST_VLM_KEY",
                    },
                }
            }
        ).vla
        scheduler = build_http_vla_scheduler(
            settings,
            environment={"TEST_VLM_KEY": "integration-secret"},
        )
        try:
            disposition = scheduler.submit(
                RGBFrame(
                    timestamp_s=10.0,
                    width=2,
                    height=1,
                    rgb_bytes=b"\xff\x00\x00\x00\xff\x00",
                ),
                now_s=10.0,
                request_id="integration-request-1",
            )
            completion = _completion(scheduler)
        finally:
            scheduler.close()

    assert disposition is InferenceSubmitDisposition.STARTED
    assert completion.result.requested_command.action is HighLevelAction.KEEP_LANE
    assert completion.result.response.model_id == "loopback-vlm"
    assert observed["requests"] == 1
    assert observed["authorization"] == "Bearer integration-secret"
    assert observed["request_id"] == "integration-request-1"
    request_payload = json.loads(observed["body"])
    image_url = request_payload["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_probe_artifacts_match_the_real_http_request(tmp_path) -> None:
    command_text = json.dumps(
        {
            "scene_summary": "Clear road.",
            "relevant_hazards": [],
            "meta_action": "KEEP_LANE",
            "target_speed_mps": 20.0,
            "confidence": 0.9,
            "brief_justification": "No hazards visible.",
        }
    )
    response = json.dumps(
        {"choices": [{"message": {"content": command_text}}]}
    ).encode()
    frame = RGBFrame(
        timestamp_s=10.0,
        width=2,
        height=1,
        rgb_bytes=b"\xff\x00\x00\x00\xff\x00",
    )
    scenario = ProbeScenario(
        scenario_id="http-probe",
        description="HTTP integration fixture.",
        expected_visual="Two test pixels.",
        map_name="S",
    )

    with _http_server(response_body=response) as (endpoint, observed):
        config = config_from_dict(
            {
                "vla": {
                    "enabled": True,
                    "http": {"endpoint_url": endpoint, "model_id": "probe-vlm"},
                }
            }
        )
        provider = build_http_vla_provider(config.vla)
        summary = run_vla_probe_catalog(
            config,
            [scenario],
            tmp_path / "probe",
            infer=True,
            provider=provider,
            capture=lambda _config, _scenario: ProbeCapture(frame, 10.0),
        )

    artifact_dir = tmp_path / "probe" / "http-probe"
    request_payload = json.loads(observed["body"])
    content = request_payload["messages"][0]["content"]
    image_url = content[1]["image_url"]["url"]
    assert summary.successful is True
    assert content[0]["text"] == (artifact_dir / "prompt.txt").read_text()
    assert base64.b64decode(image_url.removeprefix("data:image/png;base64,")) == (
        artifact_dir / "frame.png"
    ).read_bytes()
    assert (artifact_dir / "raw_response.txt").read_text() == command_text


def test_real_http_transport_does_not_follow_redirects_with_credentials() -> None:
    with _http_server() as (target_url, target_observed):
        with _http_server(redirect_url=target_url) as (redirect_url, redirect_observed):
            provider = OpenAICompatibleHTTPProvider(
                redirect_url,
                "local-vlm",
                api_key="redirect-secret",
            )
            request = ModelRequest(
                request_id="redirect-request",
                prompt="Return JSON.",
                frame=RGBFrame(
                    timestamp_s=1.0,
                    width=1,
                    height=1,
                    rgb_bytes=b"\x01\x02\x03",
                ),
                created_at_s=1.0,
            )

            with pytest.raises(ModelProviderError, match="HTTP 307"):
                provider.generate(request, timeout_s=1.0)

    assert redirect_observed["requests"] == 1
    assert target_observed["requests"] == 0

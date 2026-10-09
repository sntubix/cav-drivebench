from __future__ import annotations

import base64
import json
import socket
import struct
import urllib.error
import zlib
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.overlay import ASSIGNMENT_2, OverlayError, apply_overlay
from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla_runtime import build_http_vla_provider
from metadrive_starter.vla.http_provider import (
    HTTPTransport,
    HTTPTransportResponse,
    OpenAICompatibleHTTPProvider,
    encode_rgb_frame_png,
)
from metadrive_starter.vla.provider import (
    ModelFailureCategory,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelTimeoutError,
)


def _request(request_id: str = "camera-request-7") -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        prompt="Return exactly one safe command as JSON.",
        frame=RGBFrame(
            timestamp_s=12.0,
            width=2,
            height=2,
            rgb_bytes=bytes(
                [
                    255,
                    0,
                    0,
                    0,
                    255,
                    0,
                    0,
                    0,
                    255,
                    255,
                    255,
                    255,
                ]
            ),
        ),
        created_at_s=12.1,
    )


def _success(content: object = '{"action":"KEEP_LANE"}') -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


@dataclass
class RecordingTransport:
    response: object = HTTPTransportResponse(status_code=200, body=_success())
    error: BaseException | None = None

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
        max_response_bytes: int,
    ) -> HTTPTransportResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": body,
                "timeout_s": timeout_s,
                "max_response_bytes": max_response_bytes,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response  # type: ignore[return-value]


def _extract_png(request_body: bytes) -> bytes:
    payload = json.loads(request_body)
    url = payload["messages"][0]["content"][1]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    return base64.b64decode(url[len(prefix) :], validate=True)


def _decode_unfiltered_rgb_png(png: bytes) -> tuple[int, int, bytes]:
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind = png[offset + 4 : offset + 8]
        data = png[offset + 8 : offset + 8 + length]
        crc = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + data) & 0xFFFFFFFF
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            assert (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    scanlines = zlib.decompress(compressed)
    row_length = width * 3
    pixels = b"".join(
        scanlines[start + 1 : start + 1 + row_length]
        for start in range(0, len(scanlines), row_length + 1)
        if scanlines[start] == 0
    )
    return width, height, pixels


def test_provider_builds_chat_completion_with_lossless_png_and_correlation() -> None:
    transport = RecordingTransport()
    provider = OpenAICompatibleHTTPProvider(
        "https://models.example/v1/chat/completions",
        "vision-model",
        api_key="top-secret-key",
        transport=transport,
    )

    response = provider.generate(_request(), timeout_s=3.5)

    assert response.request_id == "camera-request-7"
    assert response.model_id == "vision-model"
    assert response.text == '{"action":"KEEP_LANE"}'
    assert response.latency_s >= 0.0
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "https://models.example/v1/chat/completions"
    assert call["timeout_s"] == 3.5
    assert call["max_response_bytes"] == 1024 * 1024
    assert call["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Request-ID": "camera-request-7",
        "Authorization": "Bearer top-secret-key",
    }
    payload = json.loads(call["body"])
    assert payload["model"] == "vision-model"
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.0
    assert "response_format" not in payload
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0] == {
        "type": "text",
        "text": "Return exactly one safe command as JSON.",
    }
    png = _extract_png(call["body"])
    width, height, rgb = _decode_unfiltered_rgb_png(png)
    assert (width, height, rgb) == (2, 2, _request().frame.rgb_bytes)


def test_provider_preserves_openai_response_identity_model_and_usage() -> None:
    transport = RecordingTransport(
        response=HTTPTransportResponse(
            status_code=200,
            body=json.dumps(
                {
                    "id": "chatcmpl-1",
                    "model": "effective-model-001",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                    "choices": [{"message": {"content": "{}"}}],
                }
            ).encode(),
        )
    )
    provider = OpenAICompatibleHTTPProvider(
        "https://models.example/v1/chat/completions",
        "requested-model",
        transport=transport,
    )

    response = provider.generate(_request(), timeout_s=1.0)

    assert response.metadata.provider == "openai_compatible_http"
    assert response.metadata.response_id == "chatcmpl-1"
    assert response.metadata.model_version == "effective-model-001"
    assert response.metadata.input_tokens == 100
    assert response.metadata.output_tokens == 20
    assert response.metadata.total_tokens == 120


def test_provider_preserves_preencoded_png_bytes_for_exact_replay() -> None:
    transport = RecordingTransport()
    provider = OpenAICompatibleHTTPProvider(
        "https://models.example/v1/chat/completions",
        "vision-model",
        transport=transport,
    )
    encoded = encode_rgb_frame_png(
        RGBFrame(timestamp_s=0.0, width=1, height=1, rgb_bytes=b"\x09\x08\x07")
    )
    original = _request()
    request = ModelRequest(
        request_id=original.request_id,
        prompt=original.prompt,
        frame=original.frame,
        created_at_s=original.created_at_s,
        encoded_png_bytes=encoded,
    )

    provider.generate(request, timeout_s=1.0)

    assert _extract_png(transport.calls[0]["body"]) == encoded


def test_provider_can_request_strict_vla_json_schema() -> None:
    transport = RecordingTransport()
    provider = OpenAICompatibleHTTPProvider(
        "https://models.example/v1/chat/completions",
        "vision-model",
        structured_output=True,
        transport=transport,
    )

    provider.generate(_request(), timeout_s=1.0)

    payload = json.loads(transport.calls[0]["body"])
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert response_format["json_schema"]["name"] == "vla_assessment"
    assert schema["properties"]["meta_action"]["enum"] == [
        "KEEP_LANE",
        "FOLLOW",
        "SLOW_DOWN",
        "STOP",
        "YIELD",
        "CHANGE_LANE_LEFT",
        "CHANGE_LANE_RIGHT",
        "REQUEST_FALLBACK",
    ]
    assert schema["properties"]["scene_summary"]["maxLength"] == 256
    assert schema["properties"]["brief_justification"]["maxLength"] == 256
    assert schema["properties"]["relevant_hazards"]["maxItems"] == 4
    assert set(schema["required"]) == {
        "scene_summary",
        "relevant_hazards",
        "meta_action",
        "target_speed_mps",
        "confidence",
        "brief_justification",
    }
    assert "issued_at_s" not in schema["properties"]
    assert "action_horizon_s" not in schema["properties"]


def test_provider_rejects_non_boolean_structured_output() -> None:
    with pytest.raises(ValueError, match="structured_output"):
        OpenAICompatibleHTTPProvider(
            "https://models.example/v1/chat/completions",
            "vision-model",
            structured_output="yes",  # type: ignore[arg-type]
        )


def test_provider_omits_authorization_without_api_key_and_supports_http() -> None:
    transport = RecordingTransport()
    provider = OpenAICompatibleHTTPProvider(
        "http://127.0.0.1:8000/v1/chat/completions",
        "local-vlm",
        transport=transport,
    )

    provider.generate(_request(), timeout_s=1.0)

    assert "Authorization" not in transport.calls[0]["headers"]
    assert isinstance(provider, ModelProvider)
    assert isinstance(transport, HTTPTransport)


def test_provider_requires_opt_in_for_non_loopback_plain_http() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleHTTPProvider("http://models.example/v1", "vlm")

    provider = OpenAICompatibleHTTPProvider(
        "http://host.docker.internal:8000/v1/chat/completions",
        "local-vlm",
        allow_insecure_http=True,
        transport=RecordingTransport(),
    )

    assert provider.endpoint_url.startswith("http://host.docker.internal")


def test_provider_accepts_text_block_list_content() -> None:
    transport = RecordingTransport(
        response=HTTPTransportResponse(
            status_code=200,
            body=_success(
                [
                    {"type": "text", "text": '{"action":'},
                    {"type": "output_text", "text": '"STOP"}'},
                ]
            ),
        )
    )
    provider = OpenAICompatibleHTTPProvider("https://example.test/v1", "vlm", transport=transport)

    response = provider.generate(_request(), timeout_s=1.0)

    assert response.text == '{"action":\n"STOP"}'


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "models.example/v1",
        "ftp://models.example/v1",
        "https:///v1",
        "https://bad host/v1",
        "https://user:password@models.example/v1",
        "https://models.example/v1#fragment",
        "https://models.example/v1\nInjected",
        "https://models.example/a path",
        "http://models.example:99999/v1",
    ],
)
def test_provider_rejects_invalid_endpoint_urls(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint_url"):
        OpenAICompatibleHTTPProvider(endpoint, "vlm")


@pytest.mark.parametrize("model_id", ["", "   ", None, 7])
def test_provider_rejects_invalid_model_id(model_id: object) -> None:
    with pytest.raises(ValueError, match="model_id"):
        OpenAICompatibleHTTPProvider("https://example.test/v1", model_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_provider_rejects_invalid_response_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="max_response_bytes"):
        OpenAICompatibleHTTPProvider(
            "https://example.test/v1", "vlm", max_response_bytes=limit  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_provider_rejects_invalid_max_tokens(max_tokens: object) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        OpenAICompatibleHTTPProvider(
            "https://example.test/v1",
            "vlm",
            max_tokens=max_tokens,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("timeout", [0.0, -1.0, True, float("nan"), float("inf")])
def test_provider_rejects_invalid_timeout(timeout: float) -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1", "vlm", transport=RecordingTransport()
    )
    with pytest.raises(ValueError, match="timeout_s"):
        provider.generate(_request(), timeout_s=timeout)


def test_api_key_is_redacted_from_repr_and_transport_errors() -> None:
    key = "secret-that-must-not-leak"
    transport = RecordingTransport(error=ModelProviderError(f"server echoed {key}"))
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1", "vlm", api_key=key, transport=transport
    )

    assert key not in repr(provider)
    with pytest.raises(ModelProviderError) as caught:
        provider.generate(_request(), timeout_s=1.0)
    assert key not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_api_key_is_redacted_from_timeout_errors() -> None:
    key = "timeout-secret"
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1",
        "vlm",
        api_key=key,
        transport=RecordingTransport(error=ModelTimeoutError(f"late: {key}")),
    )

    with pytest.raises(ModelTimeoutError) as caught:
        provider.generate(_request(), timeout_s=1.0)
    assert key not in str(caught.value)


def test_api_key_rejects_header_control_characters() -> None:
    with pytest.raises(ValueError, match="control characters"):
        OpenAICompatibleHTTPProvider(
            "https://example.test/v1",
            "vlm",
            api_key="secret\nInjected: value",
        )


def test_repr_redacts_endpoint_query_parameters() -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1/chat/completions?api-version=secret-version",
        "vlm",
    )

    assert "secret-version" not in repr(provider)
    assert "redacted" in repr(provider)


@pytest.mark.parametrize(
    "error",
    [
        socket.timeout("late"),
        TimeoutError("late"),
        urllib.error.URLError(socket.timeout("late")),
    ],
)
def test_transport_timeouts_are_mapped_to_model_timeout(error: BaseException) -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1",
        "vlm",
        transport=RecordingTransport(error=error),
    )

    with pytest.raises(ModelTimeoutError, match="timed out"):
        provider.generate(_request(), timeout_s=1.0)


def test_other_transport_failures_are_mapped_to_provider_error() -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1",
        "vlm",
        transport=RecordingTransport(error=OSError("connection refused")),
    )

    with pytest.raises(ModelProviderError, match="request failed"):
        provider.generate(_request(), timeout_s=1.0)


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_http_status_and_json_error_body_are_reported(status: int) -> None:
    transport = RecordingTransport(
        response=HTTPTransportResponse(
            status_code=status,
            body=json.dumps({"error": {"message": "model unavailable"}}).encode(),
        )
    )
    provider = OpenAICompatibleHTTPProvider("https://example.test/v1", "vlm", transport=transport)

    with pytest.raises(ModelProviderError, match=rf"HTTP {status}: model unavailable") as raised:
        provider.generate(_request(), timeout_s=1.0)

    expected = {
        400: ModelFailureCategory.TRANSPORT,
        401: ModelFailureCategory.AUTHENTICATION,
        429: ModelFailureCategory.QUOTA,
        500: ModelFailureCategory.TRANSPORT,
    }
    assert raised.value.category is expected[status]


@pytest.mark.parametrize("status", [408, 504])
def test_http_timeout_statuses_are_model_timeouts(status: int) -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1",
        "vlm",
        transport=RecordingTransport(
            response=HTTPTransportResponse(status, b"upstream deadline exceeded")
        ),
    )

    with pytest.raises(ModelTimeoutError, match=rf"HTTP {status}"):
        provider.generate(_request(), timeout_s=1.0)


def test_response_size_limit_is_enforced_even_for_injected_transport() -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1",
        "vlm",
        max_response_bytes=8,
        transport=RecordingTransport(
            response=HTTPTransportResponse(status_code=200, body=b"x" * 9)
        ),
    )

    with pytest.raises(ModelProviderError, match="exceeds 8 bytes"):
        provider.generate(_request(), timeout_s=1.0)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"\xff", "non-UTF-8"),
        (b"not-json", "malformed JSON"),
        (b"[]", "JSON object"),
        (b"{}", "no choices"),
        (b'{"choices":[]}', "no choices"),
        (b'{"choices":[3]}', "choice must be an object"),
        (b'{"choices":[{}]}', "no message object"),
        (b'{"choices":[{"message":{"content":7}}]}', "content must be text"),
        (
            b'{"choices":[{"message":{"content":[{"type":"image","url":"x"}]}}]}',
            "non-text block",
        ),
    ],
)
def test_malformed_success_responses_are_rejected(body: bytes, message: str) -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1",
        "vlm",
        transport=RecordingTransport(
            response=HTTPTransportResponse(status_code=200, body=body)
        ),
    )

    with pytest.raises(ModelProviderError, match=message):
        provider.generate(_request(), timeout_s=1.0)


@pytest.mark.parametrize(
    "response",
    [
        object(),
        HTTPTransportResponse(status_code=99, body=b"{}"),
        HTTPTransportResponse(status_code=True, body=b"{}"),
        HTTPTransportResponse(status_code=200, body="{}"),
    ],
)
def test_invalid_injected_transport_results_are_rejected(response: object) -> None:
    provider = OpenAICompatibleHTTPProvider(
        "https://example.test/v1",
        "vlm",
        transport=RecordingTransport(response=response),
    )

    with pytest.raises(ModelProviderError, match="transport returned"):
        provider.generate(_request(), timeout_s=1.0)


def test_provider_asks_for_the_most_likely_answer_by_default() -> None:
    transport = RecordingTransport()
    provider = OpenAICompatibleHTTPProvider(
        "https://models.example/v1/chat/completions",
        "vision-model",
        transport=transport,
    )

    provider.generate(_request(), timeout_s=1.0)

    payload = json.loads(transport.calls[0]["body"])
    assert payload["temperature"] == 0.0
    assert "seed" not in payload


def test_provider_samples_at_the_configured_temperature_and_seed() -> None:
    transport = RecordingTransport()
    provider = OpenAICompatibleHTTPProvider(
        "https://models.example/v1/chat/completions",
        "vision-model",
        temperature=0.7,
        seed=3,
        transport=transport,
    )

    provider.generate(_request(), timeout_s=1.0)

    payload = json.loads(transport.calls[0]["body"])
    assert (payload["temperature"], payload["seed"]) == (0.7, 3)


@pytest.mark.parametrize(
    ("http", "message"),
    [
        ({"temperature": 2.5}, "temperature must be between 0 and 2"),
        ({"temperature": -0.1}, "temperature must be between 0 and 2"),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"seed": True}, "seed must be a non-negative integer"),
    ],
)
def test_sampling_settings_are_bounded(http: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_dict({"vla": {"http": http}})


def test_sampling_settings_reach_the_provider_and_stay_instructor_owned() -> None:
    config = config_from_dict({"vla": {"enabled": True, "http": {"temperature": 0.7, "seed": 5}}})

    provider = build_http_vla_provider(config.vla)

    assert "temperature=0.7, seed=5" in repr(provider)
    with pytest.raises(OverlayError, match=r"vla\.http\.temperature is not permitted"):
        apply_overlay(config, {"vla": {"http": {"temperature": 0.0}}}, ASSIGNMENT_2)

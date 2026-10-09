from __future__ import annotations

import base64
import ipaddress
import json
import math
import socket
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Mapping, Protocol, runtime_checkable

from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.assessment import vla_assessment_json_schema
from metadrive_starter.vla.provider import (
    ModelFailureCategory,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelResponseMetadata,
    ModelTimeoutError,
)


DEFAULT_MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class HTTPTransportResponse:
    """Small transport-neutral HTTP result used by the provider boundary."""

    status_code: int
    body: bytes


@runtime_checkable
class HTTPTransport(Protocol):
    """Injectable HTTP boundary; test transports need no network access."""

    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
        max_response_bytes: int,
    ) -> HTTPTransportResponse:
        """POST one bounded request and return its status and response bytes."""
        ...


class UrllibHTTPTransport:
    """Standard-library transport for an OpenAI-compatible server.

    ``timeout_s`` is passed to urllib as its connect/socket-I/O timeout. It is
    not a forcibly cancellable wall-clock deadline if a peer continuously
    trickles bytes; production model servers should enforce their own maximum
    generation duration as well.
    """

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_s: float,
        max_response_bytes: int,
    ) -> HTTPTransportResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout_s) as response:
                response_body = _read_bounded(response, max_response_bytes)
                return HTTPTransportResponse(
                    status_code=int(response.getcode()),
                    body=response_body,
                )
        except urllib.error.HTTPError as exc:
            # HTTP status failures still carry useful, bounded provider error bodies.
            response_body = _read_bounded(exc, max_response_bytes)
            return HTTPTransportResponse(status_code=int(exc.code), body=response_body)
        except (socket.timeout, TimeoutError) as exc:
            raise ModelTimeoutError("model endpoint request timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ModelTimeoutError("model endpoint request timed out") from exc
            raise ModelProviderError("model endpoint request failed") from exc
        except (OSError, ValueError) as exc:
            raise ModelProviderError("model endpoint request failed") from exc


class OpenAICompatibleHTTPProvider:
    """Send camera-conditioned chat-completions requests over HTTP.

    The provider deliberately remains synchronous. A scheduler can run it away
    from the control loop while preserving the shared ``ModelProvider`` contract.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_id: str,
        *,
        api_key: str | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_tokens: int = 256,
        allow_insecure_http: bool = False,
        structured_output: bool = False,
        temperature: float = 0.0,
        seed: int | None = None,
        transport: HTTPTransport | None = None,
    ) -> None:
        if not isinstance(allow_insecure_http, bool):
            raise ValueError("allow_insecure_http must be a boolean")
        if not isinstance(structured_output, bool):
            raise ValueError("structured_output must be a boolean")
        self._endpoint_url = _validate_endpoint_url(
            endpoint_url,
            allow_insecure_http=allow_insecure_http,
        )
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must not be empty")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if api_key is not None and _contains_control_characters(api_key):
            raise ValueError("api_key must not contain control characters")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= temperature <= 2.0
        ):
            raise ValueError("temperature must be between 0 and 2")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        if transport is not None and not isinstance(transport, HTTPTransport):
            raise TypeError("transport must implement HTTPTransport.post()")

        self._model_id = model_id.strip()
        self._api_key = api_key.strip() if api_key is not None else None
        self._max_response_bytes = max_response_bytes
        self._max_tokens = max_tokens
        self._allow_insecure_http = allow_insecure_http
        self._structured_output = structured_output
        self._temperature = float(temperature)
        self._seed = seed
        self._transport = transport or UrllibHTTPTransport()

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_response_bytes(self) -> int:
        return self._max_response_bytes

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def structured_output(self) -> bool:
        return self._structured_output

    def __repr__(self) -> str:
        key_state = "<configured>" if self._api_key is not None else "<not configured>"
        safe_endpoint = _safe_endpoint_for_display(self._endpoint_url)
        return (
            f"{type(self).__name__}(endpoint_url={safe_endpoint!r}, "
            f"model_id={self._model_id!r}, api_key={key_state}, "
            f"max_response_bytes={self._max_response_bytes!r}, "
            f"max_tokens={self._max_tokens!r}, "
            f"allow_insecure_http={self._allow_insecure_http!r}, "
            f"structured_output={self._structured_output!r}, "
            f"temperature={self._temperature!r}, seed={self._seed!r})"
        )

    def generate(
        self,
        request: ModelRequest,
        *,
        timeout_s: float,
    ) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise ValueError("request must be a ModelRequest")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise ValueError("timeout_s must be finite and positive")

        body = _build_request_body(
            request,
            self._model_id,
            self._max_tokens,
            structured_output=self._structured_output,
            temperature=self._temperature,
            seed=self._seed,
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Request-ID": request.request_id,
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        started_at = time.monotonic()
        try:
            response = self._transport.post(
                url=self._endpoint_url,
                headers=headers,
                body=body,
                timeout_s=float(timeout_s),
                max_response_bytes=self._max_response_bytes,
            )
        except ModelTimeoutError as exc:
            raise ModelTimeoutError(
                _redact_api_key(str(exc), self._api_key)
            ) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise ModelTimeoutError("model endpoint request timed out") from exc
        except ModelProviderError as exc:
            raise ModelProviderError(
                _redact_api_key(str(exc), self._api_key)
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ModelTimeoutError("model endpoint request timed out") from exc
            raise ModelProviderError("model endpoint request failed") from exc
        except Exception as exc:
            raise ModelProviderError("model endpoint request failed") from exc
        latency_s = max(0.0, time.monotonic() - started_at)

        if not isinstance(response, HTTPTransportResponse):
            raise ModelProviderError("HTTP transport returned an invalid response")
        if (
            isinstance(response.status_code, bool)
            or not isinstance(response.status_code, int)
            or response.status_code < 100
            or response.status_code > 599
        ):
            raise ModelProviderError("HTTP transport returned an invalid status code")
        if not isinstance(response.body, bytes):
            raise ModelProviderError("HTTP transport returned a non-bytes response body")
        if len(response.body) > self._max_response_bytes:
            raise ModelProviderError(
                f"model response exceeds {self._max_response_bytes} bytes"
            )
        if response.status_code < 200 or response.status_code >= 300:
            detail = _error_detail(response.body)
            detail = _redact_api_key(detail, self._api_key)
            message = f"model endpoint returned HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            if response.status_code in (408, 504):
                raise ModelTimeoutError(message)
            raise ModelProviderError(
                message,
                category=_http_failure_category(response.status_code),
            )

        text, metadata = _decode_response_content(response.body)
        return ModelResponse(
            request_id=request.request_id,
            text=text,
            model_id=self._model_id,
            latency_s=latency_s,
            metadata=metadata,
        )


def _validate_endpoint_url(value: object, *, allow_insecure_http: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("endpoint_url must not be empty")
    endpoint_url = value.strip()
    if _contains_control_characters(endpoint_url) or any(
        character.isspace() for character in endpoint_url
    ):
        raise ValueError("endpoint_url must not contain whitespace or control characters")
    try:
        parsed = urllib.parse.urlsplit(endpoint_url)
        # Accessing port performs urllib's numeric/range validation.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint_url is not a valid HTTP(S) URL") from exc
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("endpoint_url must use HTTP or HTTPS")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError("endpoint_url must include a host")
    if any(character.isspace() for character in parsed.hostname):
        raise ValueError("endpoint_url host must not contain whitespace")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint_url must not contain credentials")
    if parsed.fragment:
        raise ValueError("endpoint_url must not contain a fragment")
    if (
        parsed.scheme.lower() == "http"
        and not allow_insecure_http
        and not _is_loopback_host(parsed.hostname)
    ):
        raise ValueError(
            "endpoint_url must use HTTPS unless HTTP is loopback or explicitly allowed"
        )
    return endpoint_url


def _build_request_body(
    request: ModelRequest,
    model_id: str,
    max_tokens: int,
    *,
    structured_output: bool,
    temperature: float = 0.0,
    seed: int | None = None,
) -> bytes:
    png_bytes = request.encoded_png_bytes or encode_rgb_frame_png(request.frame)
    image_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    if structured_output:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "vla_assessment",
                "strict": True,
                "schema": vla_assessment_json_schema(),
            },
        }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def encode_rgb_frame_png(frame: RGBFrame) -> bytes:
    """Encode tightly packed 8-bit RGB pixels as a lossless PNG."""
    if not isinstance(frame, RGBFrame):
        raise ValueError("frame must be an RGBFrame")
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", frame.width, frame.height, 8, 2, 0, 0, 0)
    row_size = frame.width * 3
    scanlines = b"".join(
        b"\x00" + frame.rgb_bytes[offset : offset + row_size]
        for offset in range(0, len(frame.rgb_bytes), row_size)
    )
    return (
        signature
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _read_bounded(stream: BinaryIO, maximum_bytes: int) -> bytes:
    body = stream.read(maximum_bytes + 1)
    if len(body) > maximum_bytes:
        raise ModelProviderError(f"model response exceeds {maximum_bytes} bytes")
    return body


def _decode_response_content(body: bytes) -> tuple[str, ModelResponseMetadata]:
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _invalid_response("model endpoint returned non-UTF-8 JSON") from exc
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise _invalid_response("model endpoint returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise _invalid_response("model endpoint response must be a JSON object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _invalid_response("model endpoint response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _invalid_response("model endpoint choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _invalid_response("model endpoint choice has no message object")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                raise _invalid_response(
                    "model endpoint content block must be an object"
                )
            if block.get("type") not in ("text", "output_text"):
                raise _invalid_response(
                    "model endpoint content contains a non-text block"
                )
            text = block.get("text")
            if not isinstance(text, str):
                raise _invalid_response(
                    "model endpoint text block has invalid text"
                )
            text_parts.append(text)
        if not text_parts:
            raise _invalid_response("model endpoint content has no text blocks")
        text = "\n".join(text_parts)
    else:
        raise _invalid_response("model endpoint message content must be text")

    usage = payload.get("usage")
    return text, ModelResponseMetadata(
        provider="openai_compatible_http",
        response_id=_optional_text(payload.get("id")),
        model_version=_optional_text(payload.get("model")),
        input_tokens=_optional_token_count(usage, "prompt_tokens"),
        output_tokens=_optional_token_count(usage, "completion_tokens"),
        total_tokens=_optional_token_count(usage, "total_tokens"),
    )


def _invalid_response(message: str) -> ModelProviderError:
    return ModelProviderError(
        message,
        category=ModelFailureCategory.INVALID_RESPONSE,
    )


def _http_failure_category(status_code: int) -> ModelFailureCategory:
    if status_code in {401, 403}:
        return ModelFailureCategory.AUTHENTICATION
    if status_code == 404:
        return ModelFailureCategory.MODEL_UNAVAILABLE
    if status_code == 429:
        return ModelFailureCategory.QUOTA
    return ModelFailureCategory.TRANSPORT


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_token_count(usage: object, name: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _error_detail(body: bytes) -> str:
    if not body:
        return ""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return "non-UTF-8 error body"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _shorten(text.strip())
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return _shorten(error["message"].strip())
        if isinstance(error, str):
            return _shorten(error.strip())
        detail = payload.get("detail")
        if isinstance(detail, str):
            return _shorten(detail.strip())
    return "unrecognized error body"


def _shorten(value: str, maximum_characters: int = 300) -> str:
    if len(value) <= maximum_characters:
        return value
    return value[:maximum_characters] + "..."


def _redact_api_key(value: str, api_key: str | None) -> str:
    if api_key:
        return value.replace(api_key, "<redacted>")
    return value


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _safe_endpoint_for_display(endpoint_url: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint_url)
    if not parsed.query:
        return endpoint_url
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "<redacted>", parsed.fragment)
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: object,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None

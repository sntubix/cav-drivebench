from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class CameraCaptureError(RuntimeError):
    """Raised when the configured simulator camera cannot provide an RGB frame."""


@dataclass(frozen=True)
class RGBFrame:
    """Transport-neutral image with tightly packed row-major RGB bytes."""

    timestamp_s: float
    width: int
    height: int
    rgb_bytes: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_s, bool)
            or not isinstance(self.timestamp_s, (int, float))
            or not math.isfinite(self.timestamp_s)
            or self.timestamp_s < 0.0
        ):
            raise ValueError("timestamp_s must be finite and non-negative")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer")
        if not isinstance(self.rgb_bytes, bytes):
            raise ValueError("rgb_bytes must be immutable bytes")

        expected_size = self.width * self.height * 3
        if len(self.rgb_bytes) != expected_size:
            raise ValueError(
                f"rgb_bytes has {len(self.rgb_bytes)} bytes; expected {expected_size} "
                f"for a {self.width}x{self.height} RGB image"
            )


SensorLookup = Callable[[Any, str], Any]


class MetaDriveCameraAdapter:
    """Capture a MetaDrive 0.4.3 RGB camera as a transport-neutral frame.

    MetaDrive's ``RGBCamera.perceive`` returns BGR channel order on the CPU,
    for compatibility with OpenCV. This adapter deliberately converts that
    simulator representation to packed RGB before it reaches a model provider.
    """

    def __init__(
        self,
        sensor_id: str = "rgb_camera",
        *,
        sensor_lookup: SensorLookup | None = None,
    ) -> None:
        if not sensor_id:
            raise ValueError("sensor_id must not be empty")
        self.sensor_id = sensor_id
        self._sensor_lookup = sensor_lookup or _lookup_sensor

    def capture(self, env: Any, *, timestamp_s: float) -> RGBFrame:
        try:
            sensor = self._sensor_lookup(env, self.sensor_id)
        except Exception as exc:
            raise CameraCaptureError(
                f"MetaDrive camera sensor {self.sensor_id!r} is unavailable"
            ) from exc

        if sensor is None or not callable(getattr(sensor, "perceive", None)):
            raise CameraCaptureError(
                f"MetaDrive camera sensor {self.sensor_id!r} does not support perceive()"
            )

        try:
            image = sensor.perceive(to_float=False)
        except Exception as exc:
            raise CameraCaptureError(
                f"MetaDrive camera sensor {self.sensor_id!r} failed to capture a frame"
            ) from exc

        rgb = _metadrive_bgr_to_rgb(image)
        height, width, _ = rgb.shape
        return RGBFrame(
            timestamp_s=timestamp_s,
            width=width,
            height=height,
            rgb_bytes=rgb.tobytes(order="C"),
        )


def _lookup_sensor(env: Any, sensor_id: str) -> Any:
    try:
        engine = env.engine
    except AttributeError as exc:
        raise CameraCaptureError("MetaDrive environment has no engine") from exc
    return engine.get_sensor(sensor_id)


def _metadrive_bgr_to_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise CameraCaptureError(
            f"MetaDrive camera returned shape {array.shape}; expected (height, width, 3)"
        )
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise CameraCaptureError("MetaDrive camera returned an empty image")

    if array.dtype == np.uint8:
        pixels = array
    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise CameraCaptureError("MetaDrive camera returned non-finite pixels")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise CameraCaptureError("normalized MetaDrive camera pixels must be between 0 and 1")
        pixels = np.rint(array * 255.0).astype(np.uint8)
    else:
        raise CameraCaptureError(
            f"MetaDrive camera returned unsupported dtype {array.dtype}; "
            "expected uint8 or normalized floating point"
        )

    return np.ascontiguousarray(pixels[..., ::-1])

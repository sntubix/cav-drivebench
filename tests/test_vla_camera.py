from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metadrive_starter.vla.camera import CameraCaptureError, MetaDriveCameraAdapter, RGBFrame


class _Sensor:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.to_float: bool | None = None

    def perceive(self, *, to_float: bool) -> np.ndarray:
        self.to_float = to_float
        return self.image


def _adapter_for(sensor: object) -> MetaDriveCameraAdapter:
    return MetaDriveCameraAdapter(sensor_lookup=lambda _env, _sensor_id: sensor)


def test_rgb_frame_validates_and_preserves_packed_bytes() -> None:
    frame = RGBFrame(timestamp_s=1.25, width=2, height=1, rgb_bytes=b"\x01\x02\x03\x04\x05\x06")

    assert frame.timestamp_s == 1.25
    assert frame.rgb_bytes == b"\x01\x02\x03\x04\x05\x06"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestamp_s": float("nan")},
        {"timestamp_s": -0.1},
        {"timestamp_s": True},
        {"width": 0},
        {"width": True},
        {"height": 0},
        {"rgb_bytes": bytearray(3)},
        {"rgb_bytes": b"\x00\x00"},
    ],
)
def test_rgb_frame_rejects_invalid_metadata(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "timestamp_s": 0.0,
        "width": 1,
        "height": 1,
        "rgb_bytes": b"\x00\x00\x00",
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        RGBFrame(**values)  # type: ignore[arg-type]


def test_capture_converts_metadrive_uint8_bgr_to_packed_rgb() -> None:
    sensor = _Sensor(
        np.array(
            [
                [[30, 20, 10], [60, 50, 40]],
                [[90, 80, 70], [120, 110, 100]],
            ],
            dtype=np.uint8,
        )
    )

    frame = _adapter_for(sensor).capture(object(), timestamp_s=3.5)

    assert sensor.to_float is False
    assert frame == RGBFrame(
        timestamp_s=3.5,
        width=2,
        height=2,
        rgb_bytes=bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
    )


def test_capture_converts_normalized_float_pixels() -> None:
    sensor = _Sensor(np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32))

    frame = _adapter_for(sensor).capture(object(), timestamp_s=0.0)

    assert frame.rgb_bytes == bytes([255, 128, 0])


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
    ],
    ids=["two-dimensional", "four-channels", "empty"],
)
def test_capture_rejects_bad_image_shapes(image: np.ndarray) -> None:
    with pytest.raises(CameraCaptureError, match="shape|empty"):
        _adapter_for(_Sensor(image)).capture(object(), timestamp_s=0.0)


@pytest.mark.parametrize(
    "image",
    [
        np.array([[[0.0, 1.1, 0.0]]], dtype=np.float32),
        np.array([[[0.0, np.nan, 0.0]]], dtype=np.float32),
        np.zeros((1, 1, 3), dtype=np.int16),
    ],
    ids=["out-of-range", "non-finite", "unsupported-integer"],
)
def test_capture_rejects_invalid_pixel_values_or_types(image: np.ndarray) -> None:
    with pytest.raises(CameraCaptureError):
        _adapter_for(_Sensor(image)).capture(object(), timestamp_s=0.0)


def test_capture_uses_default_engine_sensor_lookup() -> None:
    sensor = _Sensor(np.zeros((1, 1, 3), dtype=np.uint8))
    engine = SimpleNamespace(get_sensor=lambda sensor_id: sensor if sensor_id == "front" else None)

    frame = MetaDriveCameraAdapter("front").capture(SimpleNamespace(engine=engine), timestamp_s=1.0)

    assert frame.width == 1
    assert frame.height == 1


def test_capture_reports_missing_sensor() -> None:
    def missing_sensor(_env: object, _sensor_id: str) -> object:
        raise ValueError("not registered")

    adapter = MetaDriveCameraAdapter(sensor_lookup=missing_sensor)

    with pytest.raises(CameraCaptureError, match="unavailable"):
        adapter.capture(object(), timestamp_s=0.0)


def test_capture_rejects_object_without_camera_api() -> None:
    with pytest.raises(CameraCaptureError, match=r"perceive\(\)"):
        _adapter_for(object()).capture(object(), timestamp_s=0.0)

from __future__ import annotations

from pathlib import Path

from scripts.native_render_check import frame_problems, pipe_identity, render


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _gradient(width: int, height: int) -> bytes:
    return bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in (x * 255 // width, y * 255 // height, (x + y) % 256)
    )


def test_a_rendered_scene_is_valid_and_a_frame_that_draws_nothing_is_not() -> None:
    assert frame_problems(64, 32, _gradient(64, 32), expected_width=64, expected_height=32) == []

    # A context that draws nothing returns its clear colour, often black.
    black = frame_problems(64, 32, bytes(64 * 32 * 3), expected_width=64, expected_height=32)
    assert len(black) == 2
    assert "holds 1 colours" in black[0]
    assert "luminance spread is 0.0" in black[1]

    resized = frame_problems(32, 32, _gradient(32, 32), expected_width=64, expected_height=32)
    assert resized == ["frame is 32x32; the camera is 64x32"]


class _Named:
    def __init__(self, name: str) -> None:
        self._name = name

    def getName(self) -> str:  # noqa: N802 - Panda3D's API
        return self._name


class _FakePipe:
    def getType(self) -> _Named:  # noqa: N802
        return _Named("eglGraphicsPipe")

    def getInterfaceName(self) -> str:  # noqa: N802
        return "OpenGL"


class _FakeGsg:
    def getDriverVendor(self) -> str:  # noqa: N802
        return "NVIDIA Corporation"

    def getDriverRenderer(self) -> str:  # noqa: N802
        return "NVIDIA RTX PRO 6000 Blackwell/PCIe/SSE2"

    def getDriverVersion(self) -> str:  # noqa: N802
        raise RuntimeError("no context")


class _FakeWindow:
    def getGsg(self) -> _FakeGsg:  # noqa: N802
        return _FakeGsg()

    def getType(self) -> _Named:  # noqa: N802
        return _Named("GraphicsBuffer")


class _FakeEngine:
    pipe = _FakePipe()
    win = _FakeWindow()


def test_pipe_identity_records_the_pipe_and_driver_and_never_raises() -> None:
    assert pipe_identity(_FakeEngine()) == {
        "pipe": "eglGraphicsPipe",
        "pipe_interface": "OpenGL",
        "window": "GraphicsBuffer",
        "gl_vendor": "NVIDIA Corporation",
        "gl_renderer": "NVIDIA RTX PRO 6000 Blackwell/PCIe/SSE2",
        "gl_version": "unavailable",
    }


def test_metadrive_renders_a_valid_frame_and_names_its_pipe() -> None:
    frame, identity, (width, height) = render(PROJECT_ROOT / "configs" / "default.yaml")

    assert frame_problems(
        frame.width,
        frame.height,
        frame.rgb_bytes,
        expected_width=width,
        expected_height=height,
    ) == []
    assert identity["pipe"] != "unavailable"
    assert identity["gl_renderer"] != "unavailable"

#!/usr/bin/env python3
"""Render one camera frame and record which Panda3D pipe drew it.

The native validator runs this under a private Xvfb display when Xvfb is
installed, and otherwise with no display at all, where Panda3D falls back from
GLX to its headless EGL pipe. A frame is valid when it has the configured
camera size and is not close to one flat colour, which is what a context that
draws nothing returns. Real probe frames hold about 15,000 colours with a
luminance spread near 20, far above both thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

MINIMUM_DISTINCT_COLOURS = 256
MINIMUM_LUMINANCE_STDDEV = 4.0


def frame_statistics(rgb_bytes: bytes) -> dict[str, float | int]:
    pixels = [rgb_bytes[index : index + 3] for index in range(0, len(rgb_bytes), 3)]
    luminance = [0.299 * red + 0.587 * green + 0.114 * blue for red, green, blue in pixels]
    mean = sum(luminance) / len(luminance)
    variance = sum((value - mean) ** 2 for value in luminance) / len(luminance)
    return {
        "distinct_colours": len(set(pixels)),
        "luminance_mean": round(mean, 3),
        "luminance_stddev": round(math.sqrt(variance), 3),
    }


def frame_problems(
    width: int,
    height: int,
    rgb_bytes: bytes,
    *,
    expected_width: int,
    expected_height: int,
) -> list[str]:
    if (width, height) != (expected_width, expected_height):
        return [f"frame is {width}x{height}; the camera is {expected_width}x{expected_height}"]
    if len(rgb_bytes) != width * height * 3 or not rgb_bytes:
        return ["frame does not hold one RGB triple per pixel"]
    statistics = frame_statistics(rgb_bytes)
    problems = []
    if statistics["distinct_colours"] < MINIMUM_DISTINCT_COLOURS:
        problems.append(
            f"frame holds {statistics['distinct_colours']} colours; "
            f"a rendered scene holds at least {MINIMUM_DISTINCT_COLOURS}"
        )
    if statistics["luminance_stddev"] < MINIMUM_LUMINANCE_STDDEV:
        problems.append(
            f"frame luminance spread is {statistics['luminance_stddev']}; "
            f"a rendered scene spreads at least {MINIMUM_LUMINANCE_STDDEV}"
        )
    return problems


def _text(read: Any) -> str:
    try:
        value = str(read())
    except Exception:  # noqa: BLE001 - identity is evidence, never a crash
        return "unavailable"
    return value or "unavailable"


def pipe_identity(engine: Any) -> dict[str, str]:
    """The pipe, buffer, and OpenGL driver of MetaDrive's Panda3D engine."""
    pipe = engine.pipe
    gsg = engine.win.getGsg()
    return {
        "pipe": _text(lambda: pipe.getType().getName()),
        "pipe_interface": _text(pipe.getInterfaceName),
        "window": _text(lambda: engine.win.getType().getName()),
        "gl_vendor": _text(gsg.getDriverVendor),
        "gl_renderer": _text(gsg.getDriverRenderer),
        "gl_version": _text(gsg.getDriverVersion),
    }


def render(config_path: Path) -> tuple[Any, dict[str, str], tuple[int, int]]:
    from metadrive_starter.config import load_config
    from metadrive_starter.env import make_env
    from metadrive_starter.vla import MetaDriveCameraAdapter

    config = load_config(config_path)
    camera = replace(config.camera, enabled=True)
    simulator = replace(
        config.simulator,
        map="S",
        traffic_density=0.0,
        num_scenarios=1,
        horizon=2,
        headless=True,
        manual_control=False,
    )
    env = make_env(simulator, camera)
    try:
        env.reset()
        frame = MetaDriveCameraAdapter().capture(env, timestamp_s=0.0)
        identity = pipe_identity(env.engine)
    finally:
        env.close()
    return frame, identity, (camera.width, camera.height)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-path", choices=("xvfb", "headless"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    args = parser.parse_args()

    from metadrive_starter.vla import encode_rgb_frame_png

    frame, identity, (width, height) = render(args.config)
    problems = frame_problems(
        frame.width,
        frame.height,
        frame.rgb_bytes,
        expected_width=width,
        expected_height=height,
    )
    args.frame.parent.mkdir(parents=True, exist_ok=True)
    args.frame.write_bytes(encode_rgb_frame_png(frame))
    record = {
        "schema_version": 1,
        "render_path": args.render_path,
        "display": os.environ.get("DISPLAY", ""),
        **identity,
        "width": frame.width,
        "height": frame.height,
        "rgb_sha256": hashlib.sha256(frame.rgb_bytes).hexdigest(),
        **frame_statistics(frame.rgb_bytes),
        "valid": not problems,
        "problems": problems,
        "frame_file": args.frame.name,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

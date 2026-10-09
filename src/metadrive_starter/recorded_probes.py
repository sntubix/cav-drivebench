"""Probe scenes recorded once, so the course notebooks never drive MetaDrive.

The probe captures each scene's camera frame and measures its ``LocalScene``,
then saves the observation it built from them. A recorded catalog keeps that
probe output and, beside it in ``local_scene.json``, the measured scene itself,
in the form event logs record it. From those an ``ObservationRequest`` can be
rebuilt exactly as the probe built it, so a team's ``observation.py`` sees what
``probe --submission`` would show it, without a simulator.

Probe scenes are deterministic: a scene's measured ``LocalScene`` is the same
whenever it is captured. Its frame is not: a few dozen pixels differ between
renders, even on one machine, and more between GPU vendors. Nothing here
matches frames by hash.

Re-record after any change to MetaDrive, the probe manifest, the output
contract, or scene measurement, into a new directory:

    uv run python -m metadrive_starter.recorded_probes --output-dir tmp/recorded-probes-01
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from metadrive_starter.config import AppConfig, load_config
from metadrive_starter.events import to_json_value
from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.replay import scene_from_payload
from metadrive_starter.submission import PROJECT_ROOT
from metadrive_starter.timing import times_equal
from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.observation import ObservationRequest
from metadrive_starter.vla_probe import (
    CaptureFunction,
    ProbeCapture,
    ProbeRunSummary,
    ProbeScenario,
    capture_probe_scenario,
    load_probe_scenarios,
    load_vla_probe_replay_inputs,
    run_vla_probe_catalog,
)

# All eight public probe scenes, recorded with DriveBench's original observation
# and configs/default.yaml.
RECORDED_CATALOG = PROJECT_ROOT / "fixtures" / "vla" / "probes" / "catalog-20261002-v5"
LOCAL_SCENE_FILE = "local_scene.json"


@dataclass(frozen=True)
class RecordedProbe:
    """One recorded probe scene: what the probe captured and measured."""

    scenario: ProbeScenario
    # The camera frame as captured.
    frame: RGBFrame
    # The local scene measured with the frame.
    scene: LocalScene
    ego_speed_mps: float
    # The prompt the probe saved: the observation it sent when recording.
    prompt: str
    source_dir: Path

    @property
    def scenario_id(self) -> str:
        return self.scenario.scenario_id


def load_recorded_probes(
    directory: Path | str = RECORDED_CATALOG,
    *,
    scenario_ids: Sequence[str] | None = None,
) -> tuple[RecordedProbe, ...]:
    """Load a recorded catalog, checking every frame and prompt against its hashes."""
    probes = []
    for item in load_vla_probe_replay_inputs(directory, scenario_ids=scenario_ids):
        path = item.source_dir / LOCAL_SCENE_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; a recorded catalog keeps each scene's measured "
                "LocalScene beside its probe output"
            )
        scene = scene_from_payload(json.loads(path.read_text(encoding="utf-8")))
        frame = item.request.frame
        if not times_equal(scene.timestamp_s, frame.timestamp_s):
            raise ValueError(f"{path} was not measured with its scene's frame")
        capture = item.source_metadata["capture"]
        assert isinstance(capture, dict)
        probes.append(
            RecordedProbe(
                scenario=item.scenario,
                frame=frame,
                scene=scene,
                ego_speed_mps=float(capture["ego_speed_mps"]),
                prompt=item.request.prompt,
                source_dir=item.source_dir,
            )
        )
    return tuple(probes)


def observation_request(probe: RecordedProbe, config: AppConfig) -> ObservationRequest:
    """The request ``probe --submission`` builds for this scene under ``config``."""
    return ObservationRequest(
        frame=probe.frame,
        now_s=probe.frame.timestamp_s,
        action_horizon_s=config.vla.action_horizon_s,
        scene=probe.scene,
        ego_speed_mps=probe.ego_speed_mps,
        cruise_speed_mps=config.controller.target_speed_mps,
        max_scene_objects=config.vla.max_scene_objects,
        prompt_policy=config.vla.prompt_policy,
    )


def record_probes(
    config: AppConfig,
    scenarios: Sequence[ProbeScenario],
    output_dir: Path | str,
    *,
    scenario_ids: Sequence[str] | None = None,
    capture_scenario: CaptureFunction = capture_probe_scenario,
) -> ProbeRunSummary:
    """Capture each scene as the probe does and keep its measured scene beside it."""
    scenes: dict[str, LocalScene] = {}

    def capture(config: AppConfig, scenario: ProbeScenario) -> ProbeCapture:
        captured = capture_scenario(config, scenario)
        if captured.scene is None:
            raise ValueError(f"probe scene {scenario.scenario_id!r} measured no scene")
        scenes[scenario.scenario_id] = captured.scene
        return captured

    summary = run_vla_probe_catalog(
        config,
        scenarios,
        output_dir,
        scenario_ids=scenario_ids,
        capture=capture,
    )
    for result in summary.scenarios:
        if result.status != "captured":
            raise RuntimeError(
                f"probe scene {result.scenario_id!r} was not captured: {result.error_category}"
            )
        (Path(result.artifact_dir) / LOCAL_SCENE_FILE).write_text(
            json.dumps(to_json_value(scenes[result.scenario_id]), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record probe scenes with their measured LocalScene for the course notebooks.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/vla-probe-scenarios.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", dest="scenario_ids", action="append")
    args = parser.parse_args(argv)
    summary = record_probes(
        load_config(args.config),
        load_probe_scenarios(args.manifest),
        args.output_dir,
        scenario_ids=args.scenario_ids,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from metadrive_starter.config import CameraSettings, config_from_dict
from metadrive_starter.env import control_timestep_s, make_env, simulation_time_s
from metadrive_starter.events import to_json_value
from metadrive_starter.timing import RealTimePacer


EPISODE_ARTIFACT_SCHEMA_VERSION = 1
EPISODE_PAYLOAD_FILENAME = "episode.pkl"
EPISODE_METADATA_FILENAME = "metadata.json"
_ARTIFACT_TYPE = "metadrive_episode_replay"
_SERIALIZATION = "python_pickle_protocol_5"


class EpisodeArtifactError(ValueError):
    """Raised when an episode artifact is missing, invalid, or untrusted."""


@dataclass(frozen=True)
class EpisodeArtifactInfo:
    artifact_dir: str
    payload_path: str
    metadata_path: str
    payload_sha256: str
    run_id: str
    scenario_id: str
    recorded_steps: int
    simulation_time_s: float
    metadrive_version: str


@dataclass(frozen=True)
class LoadedEpisodeArtifact:
    info: EpisodeArtifactInfo
    config: Mapping[str, object]
    episode: Mapping[str, Any]


@dataclass(frozen=True)
class EpisodeReplaySummary:
    artifact_dir: str
    payload_sha256: str
    run_id: str
    scenario_id: str
    recorded_steps: int
    replayed_steps: int
    verified_control_frames: int
    rendered: bool
    replay_done: bool

    @property
    def successful(self) -> bool:
        return self.replay_done and self.verified_control_frames > 0

    def to_dict(self) -> dict[str, object]:
        value = to_json_value(self)
        assert isinstance(value, dict)
        value["successful"] = self.successful
        return value


@dataclass(frozen=True)
class RaceReplaySummary:
    race_output_dir: str
    replays: tuple[EpisodeReplaySummary, ...]

    @property
    def successful(self) -> bool:
        return bool(self.replays) and all(replay.successful for replay in self.replays)

    def to_dict(self) -> dict[str, object]:
        value = to_json_value(self)
        assert isinstance(value, dict)
        value["successful"] = self.successful
        return value


def write_episode_artifact(
    artifact_dir: Path | str,
    episode: Mapping[str, Any],
    *,
    config: Mapping[str, object],
    run_id: str,
    scenario_id: str,
    recorded_steps: int,
    simulation_time_s: float,
) -> EpisodeArtifactInfo:
    root = Path(artifact_dir).resolve()
    if not isinstance(episode, Mapping):
        raise TypeError("episode must be a mapping")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    _validate_text(run_id, "run_id")
    _validate_text(scenario_id, "scenario_id")
    if (
        isinstance(recorded_steps, bool)
        or not isinstance(recorded_steps, int)
        or recorded_steps < 0
    ):
        raise ValueError("recorded_steps must be a non-negative integer")
    if not _non_negative_finite(simulation_time_s):
        raise ValueError("simulation_time_s must be finite and non-negative")

    root.mkdir(parents=True, exist_ok=False)
    payload = pickle.dumps(dict(episode), protocol=5)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    payload_path = root / EPISODE_PAYLOAD_FILENAME
    metadata_path = root / EPISODE_METADATA_FILENAME
    payload_path.write_bytes(payload)
    metadrive_version = _installed_metadrive_version()
    metadata = {
        "schema_version": EPISODE_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "serialization": _SERIALIZATION,
        "payload_filename": EPISODE_PAYLOAD_FILENAME,
        "payload_sha256": payload_sha256,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "recorded_steps": recorded_steps,
        "simulation_time_s": float(simulation_time_s),
        "metadrive_version": metadrive_version,
        "config": config,
    }
    primitive = to_json_value(metadata)
    assert isinstance(primitive, dict)
    metadata_path.write_text(
        json.dumps(primitive, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return EpisodeArtifactInfo(
        artifact_dir=str(root),
        payload_path=str(payload_path),
        metadata_path=str(metadata_path),
        payload_sha256=payload_sha256,
        run_id=run_id,
        scenario_id=scenario_id,
        recorded_steps=recorded_steps,
        simulation_time_s=float(simulation_time_s),
        metadrive_version=metadrive_version,
    )


def load_episode_artifact(
    artifact_dir: Path | str,
    *,
    trusted: bool = False,
) -> LoadedEpisodeArtifact:
    if not trusted:
        raise EpisodeArtifactError(
            "episode replay payloads use pickle; load only a trusted artifact "
            "produced by this runner"
        )
    root = Path(artifact_dir).resolve()
    metadata_path = root / EPISODE_METADATA_FILENAME
    payload_path = root / EPISODE_PAYLOAD_FILENAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeArtifactError(f"cannot read episode metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise EpisodeArtifactError("episode metadata must be a JSON object")
    expected = {
        "schema_version",
        "artifact_type",
        "serialization",
        "payload_filename",
        "payload_sha256",
        "run_id",
        "scenario_id",
        "recorded_steps",
        "simulation_time_s",
        "metadrive_version",
        "config",
    }
    unknown = sorted(set(metadata) - expected)
    missing = sorted(expected - set(metadata))
    if missing:
        raise EpisodeArtifactError(
            f"episode metadata is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise EpisodeArtifactError(
            f"episode metadata has unknown fields: {', '.join(unknown)}"
        )
    if metadata["schema_version"] != EPISODE_ARTIFACT_SCHEMA_VERSION:
        raise EpisodeArtifactError(
            f"episode schema version must be {EPISODE_ARTIFACT_SCHEMA_VERSION}"
        )
    if metadata["artifact_type"] != _ARTIFACT_TYPE:
        raise EpisodeArtifactError("episode artifact_type is invalid")
    if metadata["serialization"] != _SERIALIZATION:
        raise EpisodeArtifactError("episode serialization is unsupported")
    if metadata["payload_filename"] != EPISODE_PAYLOAD_FILENAME:
        raise EpisodeArtifactError("episode payload filename is invalid")
    _validate_metadata(metadata)

    try:
        payload = payload_path.read_bytes()
    except OSError as exc:
        raise EpisodeArtifactError(f"cannot read episode payload: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != metadata["payload_sha256"]:
        raise EpisodeArtifactError("episode payload SHA-256 mismatch")
    if metadata["metadrive_version"] != _installed_metadrive_version():
        raise EpisodeArtifactError(
            "episode MetaDrive version differs from installed version"
        )
    try:
        episode = pickle.loads(payload)
    except Exception as exc:
        raise EpisodeArtifactError(f"cannot decode episode payload: {exc}") from exc
    if not isinstance(episode, Mapping):
        raise EpisodeArtifactError("decoded episode must be a mapping")

    info = EpisodeArtifactInfo(
        artifact_dir=str(root),
        payload_path=str(payload_path),
        metadata_path=str(metadata_path),
        payload_sha256=actual_sha256,
        run_id=metadata["run_id"],
        scenario_id=metadata["scenario_id"],
        recorded_steps=metadata["recorded_steps"],
        simulation_time_s=float(metadata["simulation_time_s"]),
        metadrive_version=metadata["metadrive_version"],
    )
    return LoadedEpisodeArtifact(
        info=info,
        config=metadata["config"],
        episode=episode,
    )


def replay_episode_artifact(
    artifact_dir: Path | str,
    *,
    trusted: bool = False,
    render: bool = False,
    paced: bool = True,
) -> EpisodeReplaySummary:
    if not isinstance(render, bool):
        raise TypeError("render must be a boolean")
    if not isinstance(paced, bool):
        raise TypeError("paced must be a boolean")
    loaded = load_episode_artifact(artifact_dir, trusted=trusted)
    config = config_from_dict(dict(loaded.config))
    config.simulator.headless = not render
    config.simulator.manual_control = False
    config.simulator.realtime = render and paced
    config.simulator.realtime_factor = 1.0
    config.simulator.horizon = max(1, loaded.info.recorded_steps + 2)
    config.simulator.out_of_road_done = False
    config.simulator.crash_vehicle_done = False
    config.simulator.crash_object_done = False
    config.camera = CameraSettings(enabled=False)

    env = make_env(
        config.simulator,
        config.camera,
        record_episode=False,
        replay_episode=dict(loaded.episode),
    )
    pacer = RealTimePacer(realtime_factor=1.0) if render and paced else None
    replayed_steps = 0
    verified_control_frames = 0
    replay_done = False
    try:
        _, info = env.reset()
        _verify_current_replay_frame(env)
        verified_control_frames += 1
        if pacer is not None:
            pacer.reset(simulation_time_s(env))
        maximum_steps = loaded.info.recorded_steps + 1
        for _ in range(maximum_steps):
            _, _, _, _, info = env.step([0.0, 0.0])
            replayed_steps += 1
            _verify_current_replay_frame(env)
            verified_control_frames += 1
            replay_done = bool(info.get("replay_done", False))
            if render:
                env.render(
                    {
                        "mode": "recorded race replay",
                        "scenario": loaded.info.scenario_id,
                        "recorded run": loaded.info.run_id,
                        "frame": f"{replayed_steps}/{loaded.info.recorded_steps}",
                    }
                )
            if pacer is not None:
                pacer.wait(replayed_steps * control_timestep_s(config.simulator))
            if replay_done:
                break
    finally:
        env.close()

    return EpisodeReplaySummary(
        artifact_dir=loaded.info.artifact_dir,
        payload_sha256=loaded.info.payload_sha256,
        run_id=loaded.info.run_id,
        scenario_id=loaded.info.scenario_id,
        recorded_steps=loaded.info.recorded_steps,
        replayed_steps=replayed_steps,
        verified_control_frames=verified_control_frames,
        rendered=render,
        replay_done=replay_done,
    )


def replay_race_output(
    race_output_dir: Path | str,
    *,
    trusted: bool = False,
    render: bool = False,
    paced: bool = True,
) -> RaceReplaySummary:
    root = Path(race_output_dir).resolve()
    scenario_artifacts = _race_scenario_artifacts(root)
    replays = []
    for scenario_id, artifact_dir in scenario_artifacts:
        replay = replay_episode_artifact(
            artifact_dir,
            trusted=trusted,
            render=render,
            paced=paced,
        )
        if replay.scenario_id != scenario_id:
            raise EpisodeArtifactError(
                f"race replay scenario mismatch: expected {scenario_id!r}, "
                f"artifact contains {replay.scenario_id!r}"
            )
        replays.append(replay)
    return RaceReplaySummary(race_output_dir=str(root), replays=tuple(replays))


def _race_scenario_artifacts(root: Path) -> tuple[tuple[str, Path], ...]:
    summary_path = root / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeArtifactError(f"cannot read race summary: {exc}") from exc
    if not isinstance(summary, dict) or not isinstance(summary.get("runs"), list):
        raise EpisodeArtifactError("race summary must contain a runs list")

    artifacts: list[tuple[str, Path]] = []
    for index, run in enumerate(summary["runs"]):
        if not isinstance(run, dict):
            raise EpisodeArtifactError(f"race summary run {index} must be an object")
        scenario_id = run.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise EpisodeArtifactError(
                f"race summary run {index} has an invalid scenario_id"
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", scenario_id):
            raise EpisodeArtifactError(
                f"race summary scenario_id {scenario_id!r} is unsafe"
            )
        if run.get("status") != "completed":
            raise EpisodeArtifactError(
                f"race scenario {scenario_id!r} did not complete and cannot be replayed"
            )
        artifact_dir = root / scenario_id / "replay"
        if not (artifact_dir / EPISODE_METADATA_FILENAME).is_file():
            raise EpisodeArtifactError(
                f"race scenario {scenario_id!r} has no replay artifact"
            )
        artifacts.append((scenario_id, artifact_dir))
    if not artifacts:
        raise EpisodeArtifactError("race output contains no replay artifacts")
    return tuple(artifacts)


def _verify_current_replay_frame(env: object) -> None:
    engine = getattr(env, "engine", None)
    manager = getattr(engine, "replay_manager", None)
    frame = getattr(manager, "current_frame", None)
    if manager is None or frame is None:
        raise EpisodeArtifactError("MetaDrive replay frame is unavailable")
    mapping = manager.record_name_to_current_name
    if not mapping:
        raise EpisodeArtifactError("MetaDrive replay frame contains no objects")
    for recorded_name, current_name in mapping.items():
        try:
            replayed_object = manager.spawned_objects[current_name]
            recorded_state = frame.step_info[recorded_name]
        except KeyError as exc:
            raise EpisodeArtifactError(
                f"MetaDrive replay object mapping is incomplete: {exc}"
            ) from exc
        recorded_position = recorded_state.get("position")
        if not isinstance(recorded_position, (list, tuple)) or len(recorded_position) < 2:
            raise EpisodeArtifactError("recorded replay position is invalid")
        actual_x = float(replayed_object.position[0])
        actual_y = float(replayed_object.position[1])
        if math.hypot(
            actual_x - float(recorded_position[0]),
            actual_y - float(recorded_position[1]),
        ) > 1e-4:
            raise EpisodeArtifactError(
                f"replayed position differs for object {recorded_name!r}"
            )
        recorded_heading = recorded_state.get("heading_theta")
        if not isinstance(recorded_heading, (int, float)):
            raise EpisodeArtifactError("recorded replay heading is invalid")
        heading_delta = _wrap_to_pi(
            float(replayed_object.heading_theta) - float(recorded_heading)
        )
        if abs(heading_delta) > 1e-4:
            raise EpisodeArtifactError(
                f"replayed heading differs for object {recorded_name!r}"
            )


def _wrap_to_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    _validate_text(metadata["run_id"], "run_id")
    _validate_text(metadata["scenario_id"], "scenario_id")
    _validate_text(metadata["metadrive_version"], "metadrive_version")
    payload_sha256 = metadata["payload_sha256"]
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise EpisodeArtifactError("episode payload_sha256 is invalid")
    recorded_steps = metadata["recorded_steps"]
    if (
        isinstance(recorded_steps, bool)
        or not isinstance(recorded_steps, int)
        or recorded_steps < 0
    ):
        raise EpisodeArtifactError("episode recorded_steps is invalid")
    if not _non_negative_finite(metadata["simulation_time_s"]):
        raise EpisodeArtifactError("episode simulation_time_s is invalid")
    if not isinstance(metadata["config"], Mapping):
        raise EpisodeArtifactError("episode config must be an object")


def _validate_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EpisodeArtifactError(f"{name} must not be empty")


def _non_negative_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _installed_metadrive_version() -> str:
    try:
        return version("metadrive-simulator")
    except PackageNotFoundError as exc:
        raise EpisodeArtifactError("MetaDrive distribution is not installed") from exc

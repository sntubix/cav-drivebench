from __future__ import annotations

import json
from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict
from metadrive_starter.episode_replay import (
    EpisodeArtifactError,
    load_episode_artifact,
    replay_episode_artifact,
    write_episode_artifact,
)
from metadrive_starter.events import read_event_log
from metadrive_starter.simulation import run_simulation


def test_episode_artifact_round_trip_requires_explicit_trust(tmp_path) -> None:
    root = tmp_path / "episode"
    config = config_from_dict({}).to_dict()
    info = write_episode_artifact(
        root,
        {"frame": [1, 2], "map_data": {"map": "S"}},
        config=config,
        run_id="run-1",
        scenario_id="scenario-1",
        recorded_steps=2,
        simulation_time_s=0.2,
    )

    with pytest.raises(EpisodeArtifactError, match="pickle.*trusted"):
        load_episode_artifact(root)

    loaded = load_episode_artifact(root, trusted=True)

    assert loaded.info == info
    assert loaded.episode["frame"] == [1, 2]
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["payload_sha256"] == info.payload_sha256
    assert metadata["config"]["display"]["speed_unit"] == "mps"


def test_episode_artifact_rejects_payload_tampering_before_pickle_load(tmp_path) -> None:
    root = tmp_path / "episode"
    write_episode_artifact(
        root,
        {"frame": []},
        config=config_from_dict({}).to_dict(),
        run_id="run-1",
        scenario_id="scenario-1",
        recorded_steps=0,
        simulation_time_s=0.0,
    )
    payload = root / "episode.pkl"
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(EpisodeArtifactError, match="SHA-256 mismatch"):
        load_episode_artifact(root, trusted=True)


@pytest.mark.instructor
def test_headless_simulation_records_and_verifies_world_state_replay(tmp_path) -> None:
    artifact_dir = tmp_path / "recorded"
    event_log = tmp_path / "events.jsonl"
    config = config_from_dict(
        {
            "simulator": {
                "map": "S",
                "traffic_density": 0.1,
                "random_traffic": False,
                "traffic_mode": "trigger",
                "horizon": 4,
                "headless": True,
                "realtime": False,
            },
            "scenario": {"stopped_vehicle_ahead_m": 20.0},
            "event_log": {
                "enabled": True,
                "path": str(event_log),
                "scenario_id": "recorded-scenario",
            },
            "episode_recording": {
                "enabled": True,
                "path": str(artifact_dir),
            },
        }
    )

    run = run_simulation(config)
    replay = replay_episode_artifact(
        artifact_dir,
        trusted=True,
        render=False,
        paced=False,
    )

    assert run.steps == 4
    assert run.replay_artifact_path == str(artifact_dir.resolve())
    assert run.replay_artifact_sha256 == replay.payload_sha256
    assert replay.successful is True
    assert replay.recorded_steps == run.steps
    assert replay.replayed_steps == run.steps
    assert replay.verified_control_frames == run.steps + 1
    assert replay.run_id == run.run_id
    assert replay.scenario_id == "recorded-scenario"
    records = read_event_log(event_log)
    artifact_events = [
        record for record in records if record.event_type == "episode_artifact_written"
    ]
    assert len(artifact_events) == 1
    assert (
        artifact_events[0].payload["artifact"]["payload_sha256"]
        == run.replay_artifact_sha256
    )

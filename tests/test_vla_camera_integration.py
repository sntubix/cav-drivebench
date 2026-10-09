import json
import platform

import numpy as np

from metadrive_starter.config import CameraSettings, SimulatorSettings
from metadrive_starter.env import make_env
from metadrive_starter.vla import (
    HighLevelAction,
    MetaDriveCameraAdapter,
    ScriptedModelProvider,
    VLAInferencePipeline,
)


def test_metadrive_rgb_camera_produces_transport_neutral_frame() -> None:
    env = make_env(
        SimulatorSettings(map="S", traffic_density=0.0, horizon=2, headless=True),
        CameraSettings(enabled=True, width=64, height=32),
    )
    try:
        env.reset()

        frame = MetaDriveCameraAdapter().capture(env, timestamp_s=0.0)

        assert (frame.width, frame.height) == (64, 32)
        assert len(frame.rgb_bytes) == 64 * 32 * 3
        assert min(frame.rgb_bytes) < max(frame.rgb_bytes)
        if platform.system() == "Darwin":
            channel_means = np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(-1, 3).mean(axis=0)
            assert channel_means[1] >= channel_means[0] * 0.75
            assert channel_means[2] >= channel_means[0] * 0.75

        provider = ScriptedModelProvider(
            [
                json.dumps(
                    {
                        "scene_summary": "Clear road.",
                        "relevant_hazards": [],
                        "meta_action": "KEEP_LANE",
                        "target_speed_mps": 20.0,
                        "confidence": 0.9,
                        "brief_justification": "No hazard visible.",
                    }
                )
            ]
        )
        result = VLAInferencePipeline(
            provider,
            timeout_s=1.0,
            action_horizon_s=1.0,
        ).infer(frame, now_s=0.0, request_id="camera-integration")

        assert result.requested_command.action is HighLevelAction.KEEP_LANE
        assert provider.requests[0].frame is frame
    finally:
        env.close()

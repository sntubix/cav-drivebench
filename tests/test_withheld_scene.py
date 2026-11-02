"""The measured-scene fields an observation request withholds from the model (ADR-0008)."""

import json
import time
from dataclasses import fields, replace
from pathlib import Path

import pytest

from metadrive_starter.config import RequestSceneFields, config_from_dict, load_config
from metadrive_starter.overlay import ASSIGNMENT_2, OverlayError, apply_overlay
from metadrive_starter.perception import LaneRelation, LocalScene, TrackedObject
from metadrive_starter.perception.scene import TrafficLightObservation
from metadrive_starter.planning.command_executor import VLACommandExecutor
from metadrive_starter.safety import VLACommandValidator
from metadrive_starter.submission import PROJECT_ROOT, SubmissionError, apply_submission_overlay
from metadrive_starter.traffic_lights import TrafficLightState
from metadrive_starter.vla import (
    SCENE_FIELDS,
    DefaultObservationBuilder,
    ModelScene,
    Observation,
    ObservationRequest,
    RGBFrame,
    ScriptedModelProvider,
    VLAInferencePipeline,
    VLAInferenceScheduler,
    build_vla_prompt,
    build_vla_scene_context,
    withhold_scene,
)
from metadrive_starter.vla_control import VLACommandRuntime

INTERPRETATION_CONFIG = "vla-probe-qwen3-vl-2b-llama-interpretation.yaml"
ASSIGNMENT_2_WITHHELD = frozenset({"objects", "traffic_controls"})
# Every configuration that runs a team's observation for Assignment 2.
ASSIGNMENT_2_CONFIGS = (
    "demo-vla-fixture.yaml",
    "demo-vla-semantic-stopped-vehicle.yaml",
    "demo-vla-semantic-partial-occlusion.yaml",
    "vla-probe-qwen3-vl-2b-llama.yaml",
    "vla-probe-qwen3-vl-2b-llama-structured.yaml",
    "vla-probe-smolvlm-llama.yaml",
    "vla-probe-smolvlm-llama-structured.yaml",
    "demo-vla-qwen3-vl-2b-llama.yaml",
    "demo-vla-smolvlm-llama.yaml",
    "vla-probe-qwen3-vl-2b-mlx.yaml",
    "vla-probe-qwen3-vl-2b-mlx-structured.yaml",
    "vla-probe-smolvlm-mlx.yaml",
    "demo-vla-qwen3-vl-2b-mlx.yaml",
)
KEEP_LANE = json.dumps(
    {
        "scene_summary": "Clear road.",
        "relevant_hazards": [],
        "meta_action": "KEEP_LANE",
        "target_speed_mps": 8.0,
        "confidence": 0.9,
        "brief_justification": "Nothing ahead.",
    }
)


def _tracked(object_id: str, x: float, y: float, relation: LaneRelation) -> TrackedObject:
    return TrackedObject(
        object_id=object_id,
        kind="vehicle",
        relative_position_m=(x, y),
        relative_velocity_mps=(-5.0, 0.0),
        length_m=4.5,
        width_m=1.8,
        lane_relation=relation,
        in_path=relation is LaneRelation.SAME,
        path_distance_m=x if relation is LaneRelation.SAME else None,
    )


SCENE = LocalScene(
    timestamp_s=1.0,
    ego_speed_mps=5.0,
    ego_length_m=4.5,
    ego_width_m=1.8,
    lane_offset_m=0.1,
    heading_error_rad=0.02,
    objects=(
        _tracked("lead", 20.0, 0.0, LaneRelation.SAME),
        _tracked("beside", 6.0, 3.5, LaneRelation.LEFT),
    ),
    traffic_lights=(
        TrafficLightObservation(
            light_id="signal",
            state=TrafficLightState.RED,
            relative_position_m=(30.0, 0.0),
            in_path=True,
            path_distance_m=30.0,
        ),
    ),
    distance_to_left_boundary_m=5.0,
    distance_to_right_boundary_m=1.5,
    left_lane_available=True,
    right_lane_available=True,
)
FRAME = RGBFrame(1.0, 4, 2, bytes(24))


def _context(scene: LocalScene | None) -> dict:
    context = build_vla_scene_context(scene, max_scene_objects=8)
    assert isinstance(context, dict)
    return context


# The policy.


def test_by_default_a_request_carries_every_scene_field() -> None:
    assert RequestSceneFields().withheld == frozenset()
    assert config_from_dict({}).vla.request_scene_fields == RequestSceneFields()


def test_the_policy_names_exactly_the_fields_a_scene_can_withhold() -> None:
    assert tuple(item.name for item in fields(RequestSceneFields)) == SCENE_FIELDS


def test_the_policy_loads_from_configuration() -> None:
    config = config_from_dict(
        {"vla": {"request_scene_fields": {"objects": False, "traffic_controls": False}}}
    )

    assert config.vla.request_scene_fields.withheld == ASSIGNMENT_2_WITHHELD


def test_lane_clearance_cannot_be_carried_without_the_lanes() -> None:
    with pytest.raises(ValueError, match="lane_clearance needs lanes"):
        RequestSceneFields(lanes=False)

    assert RequestSceneFields(lanes=False, lane_clearance=False).withheld == {
        "lanes",
        "lane_clearance",
    }


def test_a_field_must_be_true_or_false() -> None:
    with pytest.raises(ValueError, match=r"vla\.request_scene_fields\.objects must be a boolean"):
        RequestSceneFields(objects="no")  # type: ignore[arg-type]


@pytest.mark.parametrize("filename", ASSIGNMENT_2_CONFIGS)
def test_assignment_2_withholds_objects_and_traffic_lights_and_keeps_the_lanes(
    filename: str,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / filename)

    assert config.vla.request_scene_fields.withheld == ASSIGNMENT_2_WITHHELD


def test_scene_interpretation_also_withholds_lane_clearance_and_keeps_lane_existence() -> None:
    config = load_config(PROJECT_ROOT / "configs" / INTERPRETATION_CONFIG)

    assert config.vla.request_scene_fields.withheld == ASSIGNMENT_2_WITHHELD | {"lane_clearance"}
    assert config.vla.request_scene_fields.lanes is True


def test_configurations_outside_assignment_2_keep_the_whole_scene() -> None:
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        data = path.read_text(encoding="utf-8")
        if (
            path.name in ASSIGNMENT_2_CONFIGS
            or path.name == INTERPRETATION_CONFIG
            or "\nvla:" not in f"\n{data}"
        ):
            continue
        config = load_config(path)
        assert config.vla.request_scene_fields.withheld == frozenset(), path.name


# Instructor-owned.


@pytest.mark.parametrize(
    "overlay",
    [
        {"vla": {"request_scene_fields": {"objects": True}}},
        {"vla": {"request_scene_fields": {"lanes": False, "lane_clearance": False}}},
    ],
)
def test_an_overlay_can_never_set_the_policy(overlay: dict) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml")

    with pytest.raises(OverlayError, match=r"vla\.request_scene_fields\.\w+ is not permitted"):
        apply_overlay(base, overlay, ASSIGNMENT_2)


def test_agent_yaml_cannot_give_the_model_the_withheld_scene(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "agent.yaml").write_text(
        "vla:\n  request_scene_fields:\n    objects: true\n    traffic_controls: true\n",
        encoding="utf-8",
    )
    base = load_config(PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml")

    with pytest.raises(SubmissionError, match=r"vla\.request_scene_fields\.objects is not permitted"):
        apply_submission_overlay(base, submission)


# The scene a request carries.


def test_withholding_nothing_carries_the_measured_scene_itself() -> None:
    assert withhold_scene(SCENE, ()) is SCENE
    assert withhold_scene(None, ASSIGNMENT_2_WITHHELD) is None


def test_withheld_objects_and_lights_are_left_out_never_written_as_measured_empty() -> None:
    scene = withhold_scene(SCENE, ASSIGNMENT_2_WITHHELD)

    assert isinstance(scene, ModelScene)
    assert scene.withheld == ASSIGNMENT_2_WITHHELD
    assert (scene.objects, scene.traffic_lights) == ((), ())
    assert (scene.ego_speed_mps, scene.lane_offset_m, scene.valid) == (5.0, 0.1, True)
    assert _context(scene) == {"valid": True, "lanes": _context(SCENE)["lanes"]}


def test_lane_clearance_is_judged_on_the_whole_scene_when_objects_are_withheld() -> None:
    lanes = _context(withhold_scene(SCENE, {"objects"}))["lanes"]

    assert lanes == {
        "left": {"available": True, "observed_clear": False},
        "right": {"available": True, "observed_clear": True},
    }


def test_withheld_lane_clearance_leaves_only_which_lanes_exist() -> None:
    scene = withhold_scene(SCENE, {"objects", "lane_clearance"})

    assert _context(scene)["lanes"] == {"left": {"available": True}, "right": {"available": True}}


def test_withheld_lanes_take_their_clearance_and_the_road_edges_with_them() -> None:
    scene = withhold_scene(SCENE, {"lanes"})

    assert isinstance(scene, ModelScene)
    assert scene.withheld == {"lanes", "lane_clearance"}
    assert (scene.left_lane_available, scene.right_lane_available) == (None, None)
    assert (scene.distance_to_left_boundary_m, scene.distance_to_right_boundary_m) == (None, None)
    assert set(_context(scene)) == {"valid", "traffic_controls", "objects"}


def test_withholding_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown scene fields: weather"):
        withhold_scene(SCENE, {"weather"})


def test_the_original_observation_writes_only_what_the_request_carries() -> None:
    prompt = DefaultObservationBuilder().build(
        ObservationRequest(
            frame=FRAME,
            now_s=1.0,
            action_horizon_s=2.0,
            scene=withhold_scene(SCENE, ASSIGNMENT_2_WITHHELD),
            withheld=ASSIGNMENT_2_WITHHELD,
        )
    ).prompt

    context = json.loads(prompt.split("LOCAL_SCENE_CONTEXT:\n", maxsplit=1)[1])
    assert set(context) == {"valid", "lanes"}
    assert '"lead"' not in prompt and '"signal"' not in prompt


# What the pipeline sends a builder, and what the arbiter keeps.


class _Recording:
    def __init__(self) -> None:
        self.requests: list[ObservationRequest] = []

    def build(self, request: ObservationRequest) -> Observation:
        self.requests.append(request)
        return DefaultObservationBuilder().build(request)


def _infer(withheld: frozenset[str], scene: LocalScene | None) -> ObservationRequest:
    builder = _Recording()
    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([KEEP_LANE]),
        observation_builder=builder,
        withheld_scene_fields=withheld,
    )
    pipeline.infer(
        FRAME,
        now_s=1.0,
        request_id="r-1",
        scene=scene,
        ego_speed_mps=5.0,
        cruise_speed_mps=9.72,
    )
    [request] = builder.requests
    return request


def test_without_a_policy_a_request_carries_the_measured_scene_unchanged() -> None:
    request = _infer(frozenset(), SCENE)

    assert request.scene is SCENE
    assert request.withheld == frozenset()


def test_under_assignment_2_a_request_withholds_objects_and_lights_and_keeps_the_rest() -> None:
    request = _infer(ASSIGNMENT_2_WITHHELD, SCENE)

    assert request.withheld == ASSIGNMENT_2_WITHHELD
    assert isinstance(request.scene, ModelScene)
    assert (request.scene.objects, request.scene.traffic_lights) == ((), ())
    assert (request.ego_speed_mps, request.cruise_speed_mps) == (5.0, 9.72)
    assert request.action_horizon_s == 2.0
    assert _context(request.scene)["lanes"] == _context(SCENE)["lanes"]


def test_a_request_names_what_it_withholds_when_no_scene_was_measured() -> None:
    request = _infer(ASSIGNMENT_2_WITHHELD, None)

    assert request.scene is None
    assert request.withheld == ASSIGNMENT_2_WITHHELD


def test_a_pipeline_refuses_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="withheld_scene_fields must name only"):
        VLAInferencePipeline(ScriptedModelProvider([KEEP_LANE]), withheld_scene_fields={"x"})


def test_the_unwithheld_prompt_is_the_original_one() -> None:
    request = _infer(frozenset(), SCENE)

    assert DefaultObservationBuilder().build(request).prompt == build_vla_prompt(
        now_s=1.0,
        action_horizon_s=2.0,
        scene=SCENE,
        ego_speed_mps=5.0,
        cruise_speed_mps=9.72,
    )


class _Camera:
    def capture(self, env: object, *, timestamp_s: float) -> RGBFrame:
        return RGBFrame(timestamp_s, 1, 1, b"\x01\x02\x03")


class _Arbiter:
    def __init__(self) -> None:
        self.scenes: list[LocalScene] = []

    def begin_episode(self) -> None:
        pass

    def arbitrate(self, request):
        self.scenes.append(request.scene)
        return request.assessment

    def review(self, request) -> bool:
        return True


def test_the_arbiter_judges_against_the_whole_measured_scene() -> None:
    builder = _Recording()
    arbiter = _Arbiter()
    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([KEEP_LANE]),
        observation_builder=builder,
        withheld_scene_fields=ASSIGNMENT_2_WITHHELD,
    )
    runtime = VLACommandRuntime(
        VLAInferenceScheduler(pipeline, minimum_interval_s=0.0),
        VLACommandValidator(),
        VLACommandExecutor(9.72),
        run_id="run-1",
        camera=_Camera(),  # type: ignore[arg-type]
        arbiter=arbiter,
    )
    collected = replace(SCENE, timestamp_s=1.1)
    try:
        runtime.update(object(), SCENE, now_s=1.0, ego_speed_mps=5.0, cruise_speed_mps=9.72)
        deadline = time.monotonic() + 1.0
        while not arbiter.scenes and time.monotonic() < deadline:
            runtime.update(
                object(), collected, now_s=1.1, ego_speed_mps=5.0, cruise_speed_mps=9.72
            )
            time.sleep(0.001)
    finally:
        runtime.scheduler.close()

    assert builder.requests
    assert all(isinstance(request.scene, ModelScene) for request in builder.requests)
    assert arbiter.scenes[0] is collected

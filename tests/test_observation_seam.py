import hashlib
import json
import textwrap
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from metadrive_starter import submission
from metadrive_starter.cli import main
from metadrive_starter.config import (
    AppConfig,
    EventLogSettings,
    ObservationSettings,
    SceneFieldSettings,
    config_from_dict,
    load_config,
)
from metadrive_starter.events import read_event_log
from metadrive_starter.simulation import run_simulation
from metadrive_starter.submission import (
    SubmissionError,
    apply_submission_overlay,
    load_observation,
    load_observation_for,
)
from metadrive_starter.types import ControlTick
from metadrive_starter.vla import (
    OUTPUT_CONTRACT,
    DefaultObservationBuilder,
    HighLevelAction,
    Observation,
    ObservationError,
    ObservationRequest,
    OutputContractWarning,
    RGBFrame,
    ScriptedModelProvider,
    VLAInferencePipeline,
    build_vla_prompt,
)
from metadrive_starter.vla_runtime import build_vla_scheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = PROJECT_ROOT / "configs" / "demo-vla-fixture.yaml"
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


def _fixture_config() -> AppConfig:
    """The fixture-provider demo, logging nothing, runnable from any directory."""
    config = load_config(FIXTURE_CONFIG)
    config.event_log = EventLogSettings()
    config.vla = replace(
        config.vla,
        fixture=replace(
            config.vla.fixture,
            path=str(PROJECT_ROOT / config.vla.fixture.path),
        ),
    )
    return config


def _frame(timestamp_s: float = 9.9, width: int = 4, height: int = 2) -> RGBFrame:
    return RGBFrame(timestamp_s, width, height, bytes(width * height * 3))


def _contract_prompt(*extra: str) -> str:
    return "\n".join((*OUTPUT_CONTRACT.lines, *extra))


class _FixedObservation:
    def __init__(self, prompt: str, frame: RGBFrame | None = None) -> None:
        self.prompt = prompt
        self.frame = frame

    def build(self, request: ObservationRequest) -> Observation:
        return Observation(frame=self.frame or request.frame, prompt=self.prompt)


def test_output_contract_lines_are_lines_of_the_original_prompt() -> None:
    prompt = build_vla_prompt(now_s=10.0, action_horizon_s=2.0)

    for line in OUTPUT_CONTRACT.lines:
        assert line in prompt.splitlines()
    assert OUTPUT_CONTRACT.version.endswith("drivebench-vla-assessment-v5")
    offered = OUTPUT_CONTRACT.actions.split(": ", 1)[1].split(", ")
    # No planner executes OVERTAKE or PULL_OVER, so the model is never offered them.
    assert offered == [action.value for action in HighLevelAction][:7] + ["REQUEST_FALLBACK"]
    assert "OVERTAKE" not in offered and "PULL_OVER" not in offered


def test_default_observation_is_the_captured_frame_with_the_original_prompt() -> None:
    frame = _frame()
    request = ObservationRequest(
        frame=frame,
        now_s=10.0,
        action_horizon_s=1.5,
        ego_speed_mps=4.0,
        cruise_speed_mps=9.0,
        max_scene_objects=3,
        prompt_policy="Prefer the cruise speed.",
    )

    observation = DefaultObservationBuilder().build(request)

    assert observation.frame is frame
    assert observation.prompt == build_vla_prompt(
        now_s=10.0,
        action_horizon_s=1.5,
        ego_speed_mps=4.0,
        cruise_speed_mps=9.0,
        max_scene_objects=3,
        prompt_policy="Prefer the cruise speed.",
    )


def test_pipeline_sends_the_model_exactly_the_observation_built() -> None:
    cropped = RGBFrame(9.9, 2, 1, bytes(range(6)))
    provider = ScriptedModelProvider([KEEP_LANE])
    pipeline = VLAInferencePipeline(
        provider,
        observation_builder=_FixedObservation(
            _contract_prompt("Only the road ahead matters."), cropped
        ),
    )

    result = pipeline.infer(_frame(), now_s=10.0, request_id="request-1")

    [request] = provider.requests
    assert request.prompt == _contract_prompt("Only the road ahead matters.")
    assert request.frame == cropped
    assert result.requested_command.action is HighLevelAction.KEEP_LANE


def test_builder_receives_the_captured_frame_and_the_configured_values() -> None:
    requests: list[ObservationRequest] = []

    class Recording:
        def build(self, request: ObservationRequest) -> Observation:
            requests.append(request)
            return DefaultObservationBuilder().build(request)

    frame = _frame()
    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([KEEP_LANE]),
        action_horizon_s=1.5,
        max_scene_objects=3,
        prompt_policy="Prefer the cruise speed.",
        observation_builder=Recording(),
    )

    pipeline.infer(
        frame,
        now_s=10.0,
        request_id="request-1",
        ego_speed_mps=4.0,
        cruise_speed_mps=9.0,
    )

    assert requests == [
        ObservationRequest(
            frame=frame,
            now_s=10.0,
            action_horizon_s=1.5,
            contract=OUTPUT_CONTRACT,
            ego_speed_mps=4.0,
            cruise_speed_mps=9.0,
            max_scene_objects=3,
            prompt_policy="Prefer the cruise speed.",
        )
    ]


@pytest.mark.parametrize("dropped", range(len(OUTPUT_CONTRACT.lines)))
def test_observation_without_a_contract_line_is_sent_with_a_warning(dropped: int) -> None:
    kept = [line for index, line in enumerate(OUTPUT_CONTRACT.lines) if index != dropped]
    prompt = "\n".join(["Drive well.", *kept])
    provider = ScriptedModelProvider([KEEP_LANE])
    pipeline = VLAInferencePipeline(provider, observation_builder=_FixedObservation(prompt))

    with pytest.warns(OutputContractWarning, match="lacks output-contract lines") as caught:
        pipeline.infer(_frame(), now_s=10.0, request_id="request-1")

    [warning] = caught
    assert repr(OUTPUT_CONTRACT.lines[dropped]) in str(warning.message)
    assert [request.prompt for request in provider.requests] == [prompt]
    assert (pipeline.observations_built, pipeline.observations_without_contract) == (1, 1)


def test_a_reworded_contract_line_counts_as_missing() -> None:
    prompt = _contract_prompt().replace(OUTPUT_CONTRACT.format, "Reply in JSON.")
    provider = ScriptedModelProvider([KEEP_LANE])
    pipeline = VLAInferencePipeline(provider, observation_builder=_FixedObservation(prompt))

    with pytest.warns(OutputContractWarning, match="no Markdown fences"):
        pipeline.infer(_frame(), now_s=10.0, request_id="request-1")

    assert provider.requests[0].prompt == prompt


def test_a_pipeline_warns_once_and_counts_every_observation_without_the_contract() -> None:
    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([KEEP_LANE] * 3),
        observation_builder=_FixedObservation("Describe the road, in JSON."),
    )

    with pytest.warns(OutputContractWarning) as caught:
        for number in range(3):
            pipeline.infer(_frame(), now_s=10.0, request_id=f"request-{number}")

    assert len(caught) == 1
    assert (pipeline.observations_built, pipeline.observations_without_contract) == (3, 3)


def test_every_run_in_one_process_warns_under_the_default_filter() -> None:
    # pytest.warns shows every warning. The default filter shows a message once per
    # source line, which would silence all but the first run of evaluate or race.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(2):
            pipeline = VLAInferencePipeline(
                ScriptedModelProvider([KEEP_LANE] * 2),
                observation_builder=_FixedObservation("Describe the road, in JSON."),
            )
            for number in range(2):
                pipeline.infer(_frame(), now_s=10.0, request_id=f"request-{number}")

    assert [type(warning.message) for warning in caught] == [OutputContractWarning] * 2


def test_contract_lines_may_go_anywhere_in_any_order() -> None:
    prompt = "\n".join(
        [
            "Scene notes first.",
            *(f"  {line}" for line in reversed(OUTPUT_CONTRACT.lines)),
            "DRIVING_CONTEXT:",
            "null",
        ]
    )
    provider = ScriptedModelProvider([KEEP_LANE])
    pipeline = VLAInferencePipeline(provider, observation_builder=_FixedObservation(prompt))

    with warnings.catch_warnings():
        warnings.simplefilter("error", OutputContractWarning)
        pipeline.infer(_frame(), now_s=10.0, request_id="request-1")

    assert provider.requests[0].prompt == prompt
    assert pipeline.observations_without_contract == 0


def test_observation_may_rebuild_the_image_but_never_redate_it() -> None:
    provider = ScriptedModelProvider([KEEP_LANE])
    pipeline = VLAInferencePipeline(
        provider,
        observation_builder=_FixedObservation(_contract_prompt(), _frame(timestamp_s=10.0)),
    )

    with pytest.raises(ObservationError, match="captured frame's timestamp 9.9; got 10.0"):
        pipeline.infer(_frame(timestamp_s=9.9), now_s=10.0, request_id="request-1")

    assert provider.requests == ()


def test_builder_must_return_an_observation() -> None:
    class ReturnsText:
        def build(self, request: ObservationRequest) -> object:
            return _contract_prompt()

    pipeline = VLAInferencePipeline(
        ScriptedModelProvider([KEEP_LANE]),
        observation_builder=ReturnsText(),  # type: ignore[arg-type]
    )

    with pytest.raises(ObservationError, match="must return an Observation; got a str"):
        pipeline.infer(_frame(), now_s=10.0, request_id="request-1")


@pytest.mark.parametrize(
    ("frame", "prompt", "message"),
    [
        (b"pixels", "text", "frame must be an RGBFrame; got a bytes"),
        (_frame(), "  \n", "prompt must be non-empty text"),
        (_frame(), None, "prompt must be non-empty text"),
    ],
)
def test_observation_holds_one_frame_and_its_text(
    frame: object,
    prompt: object,
    message: str,
) -> None:
    with pytest.raises(ObservationError, match=message):
        Observation(frame=frame, prompt=prompt)  # type: ignore[arg-type]


def test_pipeline_rejects_a_builder_without_build() -> None:
    with pytest.raises(TypeError, match="ObservationBuilder.build"):
        VLAInferencePipeline(
            ScriptedModelProvider([]),
            observation_builder=object(),  # type: ignore[arg-type]
        )


def test_observation_settings_parse_from_configuration() -> None:
    config = config_from_dict(
        {
            "observation": {
                "driving_context": False,
                "scene_fields": {"objects": False},
            }
        }
    )

    assert config.observation == ObservationSettings(
        driving_context=False,
        scene_context=True,
        scene_fields=SceneFieldSettings(lanes=True, traffic_controls=True, objects=False),
    )
    # Run logs and replay artifacts record the configuration as a dict.
    recorded = {"observation": config.to_dict()["observation"]}
    assert config_from_dict(recorded).observation == config.observation


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ({"scene_context": "yes"}, r"observation\.scene_context must be a boolean"),
        ({"scene_fields": {"lanes": 1}}, r"observation\.scene_fields\.lanes must be a boolean"),
    ],
)
def test_observation_settings_reject_non_boolean_channels(
    observation: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_dict({"observation": observation})


SUBMITTED_OBSERVATION = """
    from metadrive_starter.vla.observation import Observation

    class ObservationBuilder:
        def __init__(self, settings):
            self.settings = settings

        def build(self, request):
            return Observation(frame=request.frame, prompt="\\n".join(request.contract.lines))
"""

NEUTRAL_CONTROLLER = """
    class Controller:
        def __init__(self, settings):
            pass

        def update(self, tick):
            return (0.0, 0.0)

        def reset_speed_control(self):
            pass
"""

# A run that asks the model loads arbitration.py beside observation.py.
ENDORSING_ARBITER = """
    class Arbiter:
        def __init__(self, settings):
            pass

        def begin_episode(self):
            pass

        def arbitrate(self, request):
            return request.assessment

        def review(self, request):
            return True
"""


def _submission(directory: Path, **files: str) -> Path:
    for name, source in files.items():
        (directory / f"{name}.py").write_text(textwrap.dedent(source))
    return directory


def test_submitted_observation_builder_is_built_from_observation_settings(
    tmp_path: Path,
) -> None:
    factory = load_observation(_submission(tmp_path, observation=SUBMITTED_OBSERVATION))
    settings = ObservationSettings(scene_context=False)

    builder = factory(settings)
    observation = builder.build(
        ObservationRequest(frame=_frame(), now_s=10.0, action_horizon_s=2.0)
    )

    assert builder.settings is settings
    assert observation.prompt == _contract_prompt()


def test_missing_observation_file_names_the_expected_builder(tmp_path: Path) -> None:
    with pytest.raises(
        SubmissionError,
        match=(
            r"observation\.py not found; expected a file defining "
            r"ObservationBuilder\(settings\) returning an ObservationBuilder with "
            r"build\(request\)"
        ),
    ):
        load_observation(tmp_path)


def test_observation_file_loads_only_for_a_run_that_asks_the_model(tmp_path: Path) -> None:
    broken = _submission(tmp_path, observation="raise RuntimeError('never imported')")

    assert load_observation_for(broken, config_from_dict({})) is None
    with pytest.raises(SubmissionError, match="observation.py"):
        load_observation_for(broken, config_from_dict({"vla": {"enabled": True}}))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
            class ObservationBuilder:
                def __init__(self, settings):
                    pass

                def build(self):
                    pass
            """,
            r"ObservationBuilder\.build must be callable as build\(request\); "
            r"its signature is build\(\)",
        ),
        (
            """
            class ObservationBuilder:
                def __init__(self, settings):
                    pass
            """,
            r"ObservationBuilder\(settings\) returned an ObservationBuilder without "
            r"build\(request\)",
        ),
    ],
)
def test_observation_builder_without_the_protocol_names_the_expected_method(
    tmp_path: Path,
    source: str,
    message: str,
) -> None:
    factory = load_observation(_submission(tmp_path, observation=source))

    with pytest.raises(SubmissionError, match=message):
        factory(ObservationSettings())


@pytest.mark.needs("fixtures")
def test_agent_yaml_tunes_the_prompt_text_and_channels_the_builder_receives(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent.yaml").write_text(
        "vla:\n  prompt_policy: Prefer the cruise speed.\n"
        "observation:\n  scene_context: false\n"
    )
    config = apply_submission_overlay(_fixture_config(), tmp_path)
    requests: list[ObservationRequest] = []

    class Recording:
        def __init__(self, settings: ObservationSettings) -> None:
            self.settings = settings

        def build(self, request: ObservationRequest) -> Observation:
            requests.append(request)
            return DefaultObservationBuilder().build(request)

    builder = Recording(config.observation)
    scheduler = build_vla_scheduler(config.vla, observation_builder=builder)
    try:
        scheduler.pipeline.infer(_frame(), now_s=10.0, request_id="request-1")
    finally:
        scheduler.close()

    assert builder.settings == ObservationSettings(scene_context=False)
    assert [request.prompt_policy for request in requests] == ["Prefer the cruise speed."]


def test_run_without_the_vla_subsystem_builds_no_observation() -> None:
    built: list[ObservationSettings] = []

    def factory(settings: ObservationSettings) -> _FixedObservation:
        built.append(settings)
        return _FixedObservation(_contract_prompt())

    summary = run_simulation(
        config_from_dict({}),
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
        observation_factory=factory,
    )

    assert built == []
    assert summary.observation is None


def test_vla_run_builds_its_observations_with_the_injected_builder() -> None:
    built: list[ObservationSettings] = []

    def factory(settings: ObservationSettings) -> _FixedObservation:
        built.append(settings)
        return _FixedObservation(_contract_prompt())

    config = _fixture_config()
    config.observation = ObservationSettings(driving_context=False)

    summary = run_simulation(
        config,
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
        observation_factory=factory,
    )

    assert built == [config.observation]
    assert summary.observation == f"{__name__}._FixedObservation"


def test_vla_run_without_a_submitted_observation_builds_the_original_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A student release has no course/instructor/, and needs none for this.
    monkeypatch.setattr(submission, "INSTRUCTOR_DIR", tmp_path / "absent")

    summary = run_simulation(
        _fixture_config(),
        dry_run=True,
        controller_factory=lambda settings: _NeutralController(),
    )

    assert summary.observation == "metadrive_starter.vla.prompting.DefaultObservationBuilder"


def test_observation_settings_without_a_submitted_observation_are_refused() -> None:
    config = _fixture_config()
    config.observation = ObservationSettings(scene_context=False)

    with pytest.raises(ValueError, match="read by a submitted observation.py"):
        run_simulation(
            config,
            dry_run=True,
            controller_factory=lambda settings: _NeutralController(),
        )


class _NeutralController:
    def update(self, tick: ControlTick) -> tuple[float, float]:
        return (0.0, 0.0)

    def reset_speed_control(self) -> None:
        pass


def _fixture_dry_run(team: Path, tmp_path: Path) -> list[str]:
    return [
        "run",
        "--dry-run",
        "--config",
        str(FIXTURE_CONFIG),
        "--submission",
        str(team),
        "--event-log",
        str(tmp_path / "events.jsonl"),
    ]


def test_cli_run_builds_observations_with_the_submission(capsys, tmp_path: Path) -> None:
    team = _submission(
        tmp_path,
        controller=NEUTRAL_CONTROLLER,
        observation=SUBMITTED_OBSERVATION,
        arbitration=ENDORSING_ARBITER,
    )

    status = main(_fixture_dry_run(team, tmp_path))

    assert status == 0
    assert '"observation": "submission.observation.ObservationBuilder"' in (
        capsys.readouterr().out
    )


def test_cli_vla_run_needs_the_submitted_observation_file(capsys, tmp_path: Path) -> None:
    team = _submission(tmp_path, controller=NEUTRAL_CONTROLLER)

    status = main(_fixture_dry_run(team, tmp_path))

    assert status == 1
    assert "observation.py not found" in capsys.readouterr().err


def test_cli_run_without_the_model_needs_no_observation_file(capsys, tmp_path: Path) -> None:
    team = _submission(tmp_path, controller=NEUTRAL_CONTROLLER)

    assert main(["run", "--dry-run", "--submission", str(team)]) == 0
    assert '"observation": null' in capsys.readouterr().out


# Drawn by the submitted builder below: a 4x2 image, solid mid-grey.
MARKED_FRAME_RGB = bytes([128]) * (4 * 2 * 3)

MARKING_OBSERVATION = """
    from metadrive_starter.vla.camera import RGBFrame
    from metadrive_starter.vla.observation import Observation

    class ObservationBuilder:
        def __init__(self, settings):
            pass

        def build(self, request):
            frame = RGBFrame(request.frame.timestamp_s, 4, 2, bytes([128]) * (4 * 2 * 3))
            prompt = "\\n".join(["Observed by the team.", *request.contract.lines])
            return Observation(frame=frame, prompt=prompt)
"""


@pytest.mark.needs("fixtures")
def test_submitted_observation_is_what_the_model_receives_in_a_closed_loop_run(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    config.simulator.horizon = 12
    config.event_log = EventLogSettings(
        enabled=True,
        path=str(tmp_path / "events.jsonl"),
        scenario_id="observation-seam",
    )
    team = _submission(tmp_path, observation=MARKING_OBSERVATION)

    summary = run_simulation(
        config,
        controller_factory=lambda settings: _NeutralController(),
        observation_factory=load_observation(team),
    )

    completions = [
        record.payload
        for record in read_event_log(config.event_log.path)
        if record.event_type == "inference_completed"
    ]
    prompt = "\n".join(["Observed by the team.", *OUTPUT_CONTRACT.lines])
    assert summary.observation == "submission.observation.ObservationBuilder"
    assert completions
    for payload in completions:
        assert payload["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert payload["rgb_sha256"] == hashlib.sha256(MARKED_FRAME_RGB).hexdigest()
    assert summary.vla_metrics.observations_built >= len(completions)
    assert summary.vla_metrics.observations_without_output_contract == 0


@pytest.mark.needs("fixtures")
def test_closed_loop_run_sends_observations_without_the_contract_and_says_so(
    tmp_path: Path,
) -> None:
    config = _fixture_config()
    config.simulator.horizon = 12
    source = MARKING_OBSERVATION.replace(
        '["Observed by the team.", *request.contract.lines]',
        '["Describe the road as one JSON object."]',
    )
    team = _submission(tmp_path, observation=source)

    with pytest.warns(OutputContractWarning, match="lacks output-contract lines"):
        summary = run_simulation(
            config,
            controller_factory=lambda settings: _NeutralController(),
            observation_factory=load_observation(team),
        )

    metrics = summary.vla_metrics
    assert metrics.observations_built > 0
    assert metrics.observations_without_output_contract == metrics.observations_built
    # The fixture answers whatever it is asked, so every observation was sent.
    assert metrics.responses_succeeded > 0

"""Assignment 2: what the model sees.

DriveBench loads this file when you pass ``--submission`` to a run that asks
the model, for example:

    uv run metadrive-starter run --config configs/demo-vla-fixture.yaml --submission submission

Each run builds one ``ObservationBuilder(settings)``, where ``settings`` carries
the ``observation`` values from configuration. For every model request,
DriveBench calls ``build(request)`` away from the control loop and sends the
model exactly the ``Observation`` it returns: one image and the text with it.

As shipped, this is DriveBench's original observation less one line, which a
TODO replaces: the camera frame as rendered, fixed instructions, and the
measured driving and scene context as JSON. Where each fact goes is yours to
decide: described in the text, drawn into the image, or left out for the model
to read from the pixels. Each place to change is marked TODO.

The request carries only the parts of the measured scene that the course's
configuration lets the model have: ``request.withheld`` names the ones it
leaves out, and ``request.scene`` lacks them. Assignment 2 withholds the
tracked objects and the traffic lights, so the model has to find vehicles,
obstacles, and lights in the image. Your arbiter still sees the whole measured
scene, and can check the model's answer against it.

One rule holds whatever you build: keep the frame's ``timestamp_s``. Crop,
scale, or draw on the image, but never re-date it.

One recommendation is strong: keep every line of ``request.contract``
verbatim, each on a line of its own, anywhere in the prompt. They ask for
exactly what the decoder accepts. You may rephrase or replace them, but a run
that sends an observation without all of them warns you and counts it in
``vla_metrics.observations_without_output_contract``, and a model that is not
held to the schema then rarely answers in a form the decoder accepts.

``build`` runs once per request and must return promptly. The values it reads
can be tuned in agent.yaml without touching this file:

    vla:
      prompt_policy: Prefer the cruise speed on a clear lane.
    observation:
      driving_context: false  # leave out the ego and cruise speeds
      scene_fields:
        lanes: false          # leave out which lanes beside yours exist
"""

from __future__ import annotations

import json

from metadrive_starter.config import ObservationSettings
from metadrive_starter.vla.contracts import (
    VLA_ASSESSMENT_MAX_HAZARDS,
    VLA_ASSESSMENT_MAX_TEXT_CHARACTERS,
)
from metadrive_starter.vla.observation import Observation, ObservationRequest
from metadrive_starter.vla.prompting import build_vla_scene_context


class ObservationBuilder:
    """The rendered frame, fixed instructions, and the measured context as JSON."""

    def __init__(self, settings: ObservationSettings) -> None:
        self.settings = settings

    def build(self, request: ObservationRequest) -> Observation:
        settings = self.settings
        contract = request.contract
        # TODO: write the instructions your model needs. Keep each line of
        # request.contract verbatim, on a line of its own, unless you have
        # measured what rephrasing or leaving one out costs. request.model_id
        # names the model the request goes to, and SmolVLM's answers are marked
        # as well as Qwen's, so give each model the instructions it answers best.
        lines = [
            contract.version,
            "You are assessing a driving scene and proposing one high-level manoeuvre.",
            "The attached front-camera image is the PRIMARY evidence for the driving decision.",
        ]
        if settings.scene_context:
            lines.append(
                "LOCAL_SCENE_CONTEXT contains measured tactical facts. For lane availability, "
                "observed occupancy, distance, and relative speed, follow it when visual "
                "interpretation is ambiguous."
            )
        lines += [
            contract.actions,
            contract.fields,
            "Each relevant_hazards item must contain type, relative_location, and risk. "
            "Allowed types: vehicle, pedestrian, obstacle, road_feature, traffic_control, other.",
            f"Return at most {VLA_ASSESSMENT_MAX_HAZARDS} relevant hazards. "
            "Mention each observed object at most once.",
            "Allowed relative locations: front, front_left, front_right, left, right, rear, "
            "rear_left, rear_right. Allowed risks: low, medium, high.",
        ]
        if settings.driving_context:
            lines.append(
                "DRIVING_CONTEXT reports the measured ego speed and the desired clear-road "
                "cruise speed in m/s. target_speed_mps is the commanded speed for the next "
                "action horizon, not a copy of the current ego speed."
            )
        if request.prompt_policy:
            lines += ["CONFIGURED_POLICY:", request.prompt_policy]
        lines += [
            f"Propose a manoeuvre for the next {json.dumps(float(request.action_horizon_s))} "
            "seconds. The runtime owns request IDs, timestamps, and command validity; "
            "do not return those fields.",
            # TODO: ask for answers the runtime keeps, not only ones that decode.
            # Structured output holds an answer to the schema's shape, not its
            # values: a number out of range gets the answer refused, and an
            # answer below the confidence floor decodes but is discarded, so it
            # earns nothing either. A probe's metadata.json gives each refused
            # answer's reason in outcome.error_message and each decoded answer
            # in outcome.parsed_assessment; a run's summary counts the floor's
            # reasons in vla_metrics.validation_reasons.
            "scene_summary and brief_justification must each contain at most "
            f"{VLA_ASSESSMENT_MAX_TEXT_CHARACTERS} characters. Be concise and do not repeat.",
            contract.format,
        ]
        # TODO: choose which measured facts the text gives, and at what fidelity:
        # the speeds from _driving_context and the scene from _scene_context.
        if settings.driving_context:
            lines += ["DRIVING_CONTEXT:", _json(_driving_context(request))]
        if settings.scene_context:
            lines += ["LOCAL_SCENE_CONTEXT:", _json(_scene_context(request, settings))]
        # TODO: crop, scale, or draw on the image here if your observation puts
        # facts in it. Keep request.frame.timestamp_s: never re-date the frame.
        return Observation(frame=request.frame, prompt="\n".join(lines))


def _driving_context(request: ObservationRequest) -> object:
    """Measured ego speed and the clear-road cruise speed, in m/s."""
    ego_speed_mps = request.ego_speed_mps
    if ego_speed_mps is None and request.scene is not None:
        ego_speed_mps = request.scene.ego_speed_mps
    if ego_speed_mps is None and request.cruise_speed_mps is None:
        return None
    return {"ego_speed_mps": ego_speed_mps, "cruise_speed_mps": request.cruise_speed_mps}


def _scene_context(request: ObservationRequest, settings: ObservationSettings) -> object:
    """The measured scene, keeping only the fields configuration selects."""
    context = build_vla_scene_context(
        request.scene,
        max_scene_objects=request.max_scene_objects,
    )
    if not isinstance(context, dict):
        return context
    selected = {
        "lanes": settings.scene_fields.lanes,
        "traffic_controls": settings.scene_fields.traffic_controls,
        "objects": settings.scene_fields.objects,
    }
    return {name: value for name, value in context.items() if selected.get(name, True)}


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))

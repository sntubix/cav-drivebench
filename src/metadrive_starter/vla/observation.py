"""The observation seam: what the model receives for one request.

An observation is one artifact, the image and the text that accompanies it.
An ``ObservationBuilder`` assembles it from an ``ObservationRequest``; the
pipeline checks what comes back and sends the model exactly that.

The output contract is the known-good text the decoder depends on, supplied to
every builder to place, rephrase, or replace. An observation that lacks one of
its lines is still sent, but never silently: the run warns and counts it, since
without those lines a model that is not grammar-constrained rarely answers in a
form the decoder accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Protocol, TypeAlias, runtime_checkable

from metadrive_starter.perception.scene import LocalScene
from metadrive_starter.vla.camera import RGBFrame
from metadrive_starter.vla.commands import MODEL_ACTIONS
from metadrive_starter.vla.contracts import VLA_PROMPT_CONTRACT_VERSION

if TYPE_CHECKING:
    from metadrive_starter.config import ObservationSettings


class ObservationError(ValueError):
    """An observation builder returned something that cannot be sent to the model."""


class OutputContractWarning(UserWarning):
    """An observation was sent without every line of the output contract."""


@dataclass(frozen=True)
class OutputContract:
    """The prompt lines the decoder depends on.

    Recommended, not required: an observation that carries each one verbatim,
    on a line of its own and in any place, asks for exactly what the decoder
    accepts.
    """

    version: str
    actions: str
    fields: str
    format: str

    @property
    def lines(self) -> tuple[str, ...]:
        return (self.version, self.actions, self.fields, self.format)


OUTPUT_CONTRACT = OutputContract(
    version=f"Prompt contract: {VLA_PROMPT_CONTRACT_VERSION}",
    actions=(
        "Allowed actions (exact strings only): "
        f"{', '.join(action.value for action in MODEL_ACTIONS)}"
    ),
    fields=(
        "Required output fields: scene_summary, relevant_hazards, meta_action, "
        "target_speed_mps, confidence, brief_justification."
    ),
    format=(
        "Return exactly one JSON object with the required fields, "
        "with no Markdown fences or prose."
    ),
)


# The measured-scene fields an instructor may withhold from the model's
# observation, as vla.request_scene_fields names them (ADR-0008).
SCENE_FIELDS = ("lanes", "lane_clearance", "traffic_controls", "objects")


@dataclass(frozen=True)
class ModelScene(LocalScene):
    """The measured scene as an observation request carries it (ADR-0008).

    A ``LocalScene`` without the fields the instructor withholds: no objects when
    ``objects`` is withheld, no traffic lights for ``traffic_controls``, and no
    lane availability or road-edge distances for ``lanes``. ``withheld`` names
    them, so a field withheld is never mistaken for one measured empty.
    Arbitration never sees a ``ModelScene``: it judges against the whole scene.
    """

    withheld: frozenset[str] = frozenset()
    # Whether each lane beside the ego lane is observed clear, judged on the whole
    # measured scene; None when unknown or withheld.
    left_lane_clear: bool | None = None
    right_lane_clear: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.withheld, frozenset) or not self.withheld <= set(SCENE_FIELDS):
            raise ValueError(f"withheld must be a frozenset of {', '.join(SCENE_FIELDS)}")


@dataclass(frozen=True)
class ObservationRequest:
    """Everything one observation may be assembled from.

    The runtime validates every value before it builds the request.
    """

    # The camera frame as captured at now_s.
    frame: RGBFrame
    now_s: float
    # How long the proposed manoeuvre will hold, in seconds.
    action_horizon_s: float
    contract: OutputContract = OUTPUT_CONTRACT
    # The measured local scene, without the fields in withheld; None when there
    # is none.
    scene: LocalScene | None = None
    # Measured ego speed; None when unknown.
    ego_speed_mps: float | None = None
    # The desired clear-road cruise speed.
    cruise_speed_mps: float | None = None
    # The instructor bound on objects in the scene context (vla.max_scene_objects).
    max_scene_objects: int = 8
    # The configured policy text (vla.prompt_policy).
    prompt_policy: str = ""
    # The model this request goes to, as its configuration names it
    # (vla.http.model_id or vla.vertex.model_id); None when recorded answers
    # stand in for a model.
    model_id: str | None = None
    # The measured-scene fields the instructor withholds from the model
    # (vla.request_scene_fields), whether or not a scene was measured. The model
    # must find them in the image.
    withheld: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Observation:
    """What the model receives for one request: one image and the text with it."""

    frame: RGBFrame
    prompt: str

    def __post_init__(self) -> None:
        _check_parts(self.frame, self.prompt)


@runtime_checkable
class ObservationBuilder(Protocol):
    """Replaceable observation assembly: one request in, one observation out."""

    def build(self, request: ObservationRequest) -> Observation:
        """Return the image and text the model receives for ``request``.

        Called once per model request, away from the control loop. It must
        return promptly and keep the captured frame's timestamp, and should
        carry every line of ``request.contract``.
        """
        ...


ObservationBuilderFactory: TypeAlias = Callable[["ObservationSettings"], ObservationBuilder]


def build_observation_builder(
    settings: ObservationSettings,
    *,
    factory: ObservationBuilderFactory,
) -> ObservationBuilder:
    """Build an observation builder from its factory, checking it implements the protocol."""
    builder = factory(settings)
    if not isinstance(builder, ObservationBuilder):
        raise TypeError("observation builder must implement build(request)")
    return builder


def checked_observation(observation: object, request: ObservationRequest) -> Observation:
    """Return ``observation`` if it may be sent for ``request``, or explain why not.

    Only a malformed observation is refused; one that lacks output-contract
    lines is sent, and ``missing_contract_lines`` says which.
    """
    if not isinstance(observation, Observation):
        raise ObservationError(
            "observation builder build() must return an Observation; "
            f"got a {type(observation).__name__}"
        )
    _check_parts(observation.frame, observation.prompt)
    if observation.frame.timestamp_s != request.frame.timestamp_s:
        # Frame age is judged on the captured frame; the image may be rebuilt,
        # but never re-dated.
        raise ObservationError(
            "observation frame must keep the captured frame's timestamp "
            f"{request.frame.timestamp_s!r}; got {observation.frame.timestamp_s!r}"
        )
    return observation


def missing_contract_lines(prompt: str, contract: OutputContract) -> tuple[str, ...]:
    """The lines of ``contract`` that ``prompt`` does not carry verbatim on a line of its own."""
    lines = {line.strip() for line in prompt.splitlines()}
    return tuple(line for line in contract.lines if line not in lines)


def describe_missing_contract(missing: Iterable[str]) -> str:
    """Why an observation that lacks ``missing`` is likely to fail to decode."""
    return (
        f"lacks output-contract lines {_quoted(missing)}; the decoder depends on "
        "them, so expect decode failures unless the server enforces structured output"
    )


def _check_parts(frame: object, prompt: object) -> None:
    if not isinstance(frame, RGBFrame):
        raise ObservationError(
            f"observation frame must be an RGBFrame; got a {type(frame).__name__}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ObservationError("observation prompt must be non-empty text")


def _quoted(lines: Iterable[str]) -> str:
    return ", ".join(repr(line) for line in lines)

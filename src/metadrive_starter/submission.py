"""Load student-owned seam files from a submission directory.

Each seam file defines one entry point that the foundation calls with that
seam's settings; the object it returns must implement the seam's protocol.
Only files named by a seam are ever imported.

The instructor reference solutions are a submission too, kept under
course/instructor/ so that every student release leaves them out.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, cast

import yaml

from metadrive_starter.config import AppConfig
from metadrive_starter.controllers import VehicleController, VehicleControllerFactory
from metadrive_starter.overlay import ASSIGNMENT_2, Allowlist, OverlayError, apply_overlay
from metadrive_starter.vla.arbitration import AssessmentArbiter, AssessmentArbiterFactory
from metadrive_starter.vla.observation import ObservationBuilder, ObservationBuilderFactory


class SubmissionError(Exception):
    """A submitted seam file cannot be loaded, or what it builds does not
    implement the seam's protocol."""


@dataclass(frozen=True)
class SeamSpec:
    """One student-owned file and the entry point the foundation calls in it."""

    filename: str
    entry_point: str
    entry_parameters: tuple[str, ...]
    protocol: type

    @property
    def entry_call(self) -> str:
        return _call(self.entry_point, self.entry_parameters)

    @property
    def methods(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Public methods declared by the protocol, with parameters after ``self``."""
        return tuple(
            (name, tuple(inspect.signature(member).parameters)[1:])
            for name, member in vars(self.protocol).items()
            if not name.startswith("_") and callable(member)
        )

    @property
    def expectation(self) -> str:
        methods = _join(_call(name, parameters) for name, parameters in self.methods)
        return f"{self.entry_call} returning {_a(self.protocol.__name__)} with {methods}"


CONTROLLER_SEAM = SeamSpec(
    filename="controller.py",
    entry_point="Controller",
    entry_parameters=("settings",),
    protocol=VehicleController,
)

OBSERVATION_SEAM = SeamSpec(
    filename="observation.py",
    entry_point="ObservationBuilder",
    entry_parameters=("settings",),
    protocol=ObservationBuilder,
)

ARBITRATION_SEAM = SeamSpec(
    filename="arbitration.py",
    entry_point="Arbiter",
    entry_parameters=("settings",),
    protocol=AssessmentArbiter,
)

AGENT_OVERLAY = "agent.yaml"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTOR_DIR = PROJECT_ROOT / "course" / "instructor"
REFERENCE_SUBMISSION = INSTRUCTOR_DIR / "reference-submission"


def load_controller(submission_dir: Path | str) -> VehicleControllerFactory:
    """Return a factory for the ``Controller`` in ``<submission_dir>/controller.py``."""
    return cast(VehicleControllerFactory, load_seam(submission_dir, CONTROLLER_SEAM))


def load_observation(submission_dir: Path | str) -> ObservationBuilderFactory:
    """Return a factory for the ``ObservationBuilder`` in ``<submission_dir>/observation.py``."""
    return cast(ObservationBuilderFactory, load_seam(submission_dir, OBSERVATION_SEAM))


def load_observation_for(
    submission_dir: Path | str,
    config: AppConfig,
) -> ObservationBuilderFactory | None:
    """Return the observation factory a run with ``config`` needs, or None.

    ``observation.py`` loads only when the run asks the model, so a submission
    without it, or with a broken one, still drives every other run.
    """
    return load_observation(submission_dir) if config.vla.enabled else None


def load_arbitration(submission_dir: Path | str) -> AssessmentArbiterFactory:
    """Return a factory for the ``Arbiter`` in ``<submission_dir>/arbitration.py``."""
    return cast(AssessmentArbiterFactory, load_seam(submission_dir, ARBITRATION_SEAM))


def load_arbitration_for(
    submission_dir: Path | str,
    config: AppConfig,
) -> AssessmentArbiterFactory | None:
    """Return the arbiter factory a run with ``config`` needs, or None.

    ``arbitration.py`` loads only when the run asks the model, like
    ``observation.py``, so a submission without it still drives every other run.
    """
    return load_arbitration(submission_dir) if config.vla.enabled else None


def apply_submission_overlay(
    config: AppConfig,
    submission_dir: Path | str,
    *,
    allowlist: Allowlist = ASSIGNMENT_2,
) -> AppConfig:
    """Return a copy of ``config`` with ``<submission_dir>/agent.yaml`` applied.

    Without an agent.yaml the copy carries the instructor values unchanged.
    """
    path = Path(submission_dir) / AGENT_OVERLAY
    overlay: Any = {}
    # The messages below say everything; unlike an import failure, the cause's
    # traceback would point into DriveBench rather than into the team's file.
    if path.is_file():
        try:
            overlay = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise SubmissionError(f"{path} is not valid YAML: {exc}") from None
    try:
        return apply_overlay(config, overlay, allowlist)
    except OverlayError as exc:
        raise SubmissionError(f"{path}: {exc}") from None


def apply_reference_overlay(config: AppConfig) -> AppConfig:
    """Return a copy of ``config`` tuned as the instructor reference drives it.

    The reference tunes the starting gains in its own agent.yaml, as a team
    does; a release, which has no reference, gets the copy unchanged.
    """
    return apply_submission_overlay(config, REFERENCE_SUBMISSION)


def load_reference_controller() -> VehicleControllerFactory:
    """Return a factory for the instructor reference ``Controller``."""
    if not INSTRUCTOR_DIR.is_dir():
        raise SubmissionError(
            "this installation has no reference controller; "
            "run with --submission submission"
        )
    return cast(
        VehicleControllerFactory,
        load_seam(REFERENCE_SUBMISSION, CONTROLLER_SEAM, package="reference"),
    )


def load_seam(
    submission_dir: Path | str,
    spec: SeamSpec,
    *,
    package: str = "submission",
) -> Callable[..., Any]:
    """Import one seam file as ``<package>.<file stem>`` and return a factory
    for its entry point."""
    path = Path(submission_dir) / spec.filename
    if not path.is_file():
        raise SubmissionError(
            f"{path} not found; expected a file defining {spec.expectation}"
        )
    module = _import_submission(path, package)
    entry = getattr(module, spec.entry_point, None)
    if entry is None:
        raise SubmissionError(
            f"{path} must define {spec.expectation}; "
            f"it defines {_defined_names(module)}"
        )
    if not callable(entry):
        raise SubmissionError(
            f"{path}: {spec.entry_point} must be callable as {spec.entry_call}; "
            f"found {entry!r}"
        )
    _check_signature(
        entry,
        path=path,
        label=spec.entry_point,
        name=spec.entry_point,
        parameters=spec.entry_parameters,
    )

    def build(*arguments: Any) -> Any:
        try:
            product = entry(*arguments)
        except Exception as exc:
            raise SubmissionError(
                f"{path}: {spec.entry_call} raised {type(exc).__name__}: {exc}"
            ) from exc
        _check_protocol(product, path=path, spec=spec)
        return product

    return build


def _import_submission(path: Path, package: str) -> ModuleType:
    module_name = f"{package}.{path.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    # Registered like any imported module: dataclasses and typing resolve
    # postponed annotations through sys.modules[cls.__module__].
    sys.modules[module_name] = module
    try:
        # Compiled here, not by the import system, which would cache bytecode
        # inside the submission directory. dont_inherit keeps this module's
        # own __future__ flags out of the submitted code.
        code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception as exc:
        del sys.modules[module_name]
        raise SubmissionError(
            f"{path} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def _check_protocol(product: object, *, path: Path, spec: SeamSpec) -> None:
    product_type = type(product).__name__
    for name, parameters in spec.methods:
        method = getattr(product, name, None)
        if not callable(method):
            raise SubmissionError(
                f"{path}: {spec.entry_call} returned {_a(product_type)} without "
                f"{_call(name, parameters)}; expected {spec.expectation}"
            )
        _check_signature(
            method,
            path=path,
            label=f"{product_type}.{name}",
            name=name,
            parameters=parameters,
        )


def _check_signature(
    function: Callable[..., Any],
    *,
    path: Path,
    label: str,
    name: str,
    parameters: tuple[str, ...],
) -> None:
    """Require that ``function`` accepts one positional argument per parameter."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(*([None] * len(parameters)))
    except TypeError:
        bare = signature.replace(
            parameters=[
                parameter.replace(annotation=inspect.Parameter.empty)
                for parameter in signature.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
        raise SubmissionError(
            f"{path}: {label} must be callable as {_call(name, parameters)}; "
            f"its signature is {name}{bare}"
        ) from None


def _defined_names(module: ModuleType) -> str:
    names = [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__ == module.__name__
    ]
    return ", ".join(names) if names else "no classes or functions"


def _call(name: str, parameters: Iterable[str]) -> str:
    return f"{name}({', '.join(parameters)})"


def _a(noun: str) -> str:
    return f"{'an' if noun[:1] in 'AEIOU' else 'a'} {noun}"


def _join(items: Iterable[str]) -> str:
    words = list(items)
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"

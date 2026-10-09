from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from metadrive_starter.cloud_preflight import (
    VertexPreflightReport,
    run_vertex_preflight,
)
from metadrive_starter.config import AppConfig, VertexProviderSettings
from metadrive_starter.vla import (
    ModelProvider,
    RequestBudget,
    RequestBudgetError,
    RequestCappedModelProvider,
    VertexModelProvider,
)
from metadrive_starter.vla_probe import (
    ProbeRunSummary,
    load_vla_probe_replay_inputs,
    replay_vla_probe_artifacts,
)


VERTEX_CAPTURE_SCENARIO_IDS = (
    "clear-straight",
    "stopped-vehicle",
    "red-light-with-traffic",
)
VERTEX_CAPTURE_PLANNED_REQUESTS = 5
VERTEX_CAPTURE_HARD_MAXIMUM = 5

PreflightRunner = Callable[..., VertexPreflightReport]
ProviderFactory = Callable[[VertexProviderSettings, str], ModelProvider]
ProbeRunner = Callable[..., ProbeRunSummary]


@dataclass(frozen=True)
class VertexCaptureReport:
    output_dir: str
    source_dir: str
    scenario_ids: tuple[str, ...]
    request_cap: int
    planned_requests: int
    used_requests: int
    request_labels: tuple[str, ...]
    preflight: VertexPreflightReport
    probes: ProbeRunSummary | None = None

    @property
    def successful(self) -> bool:
        return (
            self.preflight.successful
            and self.probes is not None
            and self.probes.successful
            and self.used_requests == self.planned_requests
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "successful": self.successful,
            "output_dir": self.output_dir,
            "source_dir": self.source_dir,
            "scenario_ids": list(self.scenario_ids),
            "request_budget": {
                "hard_cap": self.request_cap,
                "planned": self.planned_requests,
                "used": self.used_requests,
                "remaining": self.request_cap - self.used_requests,
                "request_labels": list(self.request_labels),
                "retries": 0,
            },
            "preflight": self.preflight.to_dict(),
            "probes": self.probes.to_dict() if self.probes is not None else None,
        }


def run_vertex_capture(
    config: AppConfig,
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    project_id: str | None = None,
    vertex_settings: VertexProviderSettings | None = None,
    maximum_requests: int = VERTEX_CAPTURE_HARD_MAXIMUM,
    environment: Mapping[str, str] | None = None,
    preflight_runner: PreflightRunner = run_vertex_preflight,
    provider_factory: ProviderFactory | None = None,
    probe_runner: ProbeRunner = replay_vla_probe_artifacts,
) -> VertexCaptureReport:
    """Run the fixed live-evidence plan under one shared five-request cap."""
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if (
        isinstance(maximum_requests, bool)
        or not isinstance(maximum_requests, int)
        or maximum_requests < 1
        or maximum_requests > VERTEX_CAPTURE_HARD_MAXIMUM
    ):
        raise ValueError(
            f"maximum_requests must be between 1 and {VERTEX_CAPTURE_HARD_MAXIMUM}"
        )
    settings = vertex_settings or config.vla.vertex
    if not isinstance(settings, VertexProviderSettings):
        raise TypeError("vertex_settings must be VertexProviderSettings or None")

    budget = RequestBudget(maximum_requests)
    budget.require_capacity(VERTEX_CAPTURE_PLANNED_REQUESTS)

    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"Vertex capture output directory already exists: {root}")

    prepared_inputs = load_vla_probe_replay_inputs(
        source_dir,
        scenario_ids=VERTEX_CAPTURE_SCENARIO_IDS,
    )
    root.mkdir(parents=True)

    preflight = preflight_runner(
        settings,
        project_id=project_id,
        location=settings.location,
        model_id=settings.model_id,
        samples=1,
        request_timeout_s=config.vla.request_timeout_s,
        environment=environment,
        request_budget=budget,
        prompt_policy=config.vla.prompt_policy,
    )
    if not preflight.successful:
        report = _capture_report(root, source_dir, budget, preflight, None)
        _write_report(root / "summary.json", report)
        return report

    if preflight.project_id is None:
        raise RuntimeError("successful Vertex preflight did not resolve a project ID")
    make_provider = provider_factory or _build_vertex_provider
    provider = RequestCappedModelProvider(
        make_provider(settings, preflight.project_id),
        budget,
    )
    probes = probe_runner(
        config,
        source_dir,
        root / "probes",
        provider=provider,
        scenario_ids=VERTEX_CAPTURE_SCENARIO_IDS,
        prepared_inputs=prepared_inputs,
    )
    report = _capture_report(root, source_dir, budget, preflight, probes)
    _write_report(root / "summary.json", report)
    return report


def _build_vertex_provider(
    settings: VertexProviderSettings,
    project_id: str,
) -> VertexModelProvider:
    return VertexModelProvider(
        project_id,
        settings.location,
        settings.model_id,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        thinking_budget=settings.thinking_budget,
    )


def _capture_report(
    output_dir: Path,
    source_dir: Path | str,
    budget: RequestBudget,
    preflight: VertexPreflightReport,
    probes: ProbeRunSummary | None,
) -> VertexCaptureReport:
    return VertexCaptureReport(
        output_dir=str(output_dir),
        source_dir=str(Path(source_dir).resolve()),
        scenario_ids=VERTEX_CAPTURE_SCENARIO_IDS,
        request_cap=budget.maximum_requests,
        planned_requests=VERTEX_CAPTURE_PLANNED_REQUESTS,
        used_requests=budget.used_requests,
        request_labels=budget.request_labels,
        preflight=preflight,
        probes=probes,
    )


def _write_report(path: Path, report: VertexCaptureReport) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


__all__ = [
    "RequestBudgetError",
    "VERTEX_CAPTURE_HARD_MAXIMUM",
    "VERTEX_CAPTURE_PLANNED_REQUESTS",
    "VERTEX_CAPTURE_SCENARIO_IDS",
    "VertexCaptureReport",
    "run_vertex_capture",
]

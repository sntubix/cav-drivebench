from pathlib import Path

import pytest

from metadrive_starter.config import config_from_dict, load_config
from metadrive_starter.simulation import RunSummary, run_simulation


pytestmark = pytest.mark.instructor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_full_route(summary: RunSummary) -> None:
    assert summary.arrived
    assert summary.route_completion >= 0.95
    assert not summary.went_off_road
    assert not summary.crashed


@pytest.mark.parametrize("map_name", ["S", "C", "X", "XTOC", "O"])
def test_baseline_autopilot_completes_canonical_map(map_name: str) -> None:
    config = config_from_dict(
        {
            "simulator": {
                "map": map_name,
                "traffic_density": 0.0,
                "obstacle_probability": 0.0,
                "horizon": 1000,
                "headless": True,
            }
        }
    )

    _assert_full_route(run_simulation(config))


def test_autopilot_demo_profile_completes_full_route() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "demo-autopilot.yaml")
    config.simulator.headless = True
    config.simulator.realtime = False

    _assert_full_route(run_simulation(config))

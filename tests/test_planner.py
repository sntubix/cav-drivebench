import math

import pytest

from metadrive_starter.planning import WaypointPathPlanner
from metadrive_starter.types import EgoState, PerceptionFrame


def test_planner_selects_lookahead_target() -> None:
    planner = WaypointPathPlanner(route=[(0.0, 0.0), (5.0, 0.0), (20.0, 0.0)], lookahead_m=10.0)
    frame = PerceptionFrame(ego=EgoState(position=(0.0, 0.0), heading_rad=0.0, speed_mps=0.0))

    plan = planner.plan(frame)

    assert plan.target == (10.0, 0.0)
    assert plan.heading_error_rad == 0.0
    assert plan.distance_m == 10.0
    assert plan.lateral_error_m == 0.0


def test_planner_wraps_heading_error() -> None:
    planner = WaypointPathPlanner(route=[(-10.0, 0.0)], lookahead_m=1.0)
    frame = PerceptionFrame(ego=EgoState(position=(0.0, 0.0), heading_rad=-math.pi + 0.1, speed_mps=0.0))

    plan = planner.plan(frame)

    assert -math.pi <= plan.heading_error_rad <= math.pi


def test_planner_progress_does_not_return_to_passed_waypoints() -> None:
    planner = WaypointPathPlanner(route=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)], lookahead_m=5.0)

    planner.plan(PerceptionFrame(ego=EgoState(position=(15.0, 0.0), heading_rad=0.0, speed_mps=0.0)))
    plan = planner.plan(PerceptionFrame(ego=EgoState(position=(1.0, 0.0), heading_rad=0.0, speed_mps=0.0)))

    assert plan.target == (20.0, 0.0)


def test_planner_reports_right_positive_lateral_error() -> None:
    planner = WaypointPathPlanner(route=[(0.0, 0.0), (20.0, 0.0)], lookahead_m=5.0)

    right = planner.plan(
        PerceptionFrame(
            ego=EgoState(position=(2.0, -1.5), heading_rad=0.0, speed_mps=0.0)
        )
    )

    assert right.target == (7.0, 0.0)
    assert right.lateral_error_m == 1.5


def test_planner_uses_route_tangent_not_target_bearing_for_heading() -> None:
    planner = WaypointPathPlanner(route=[(0.0, 0.0), (20.0, 0.0)], lookahead_m=5.0)

    plan = planner.plan(
        PerceptionFrame(
            ego=EgoState(position=(2.0, 1.0), heading_rad=0.0, speed_mps=0.0)
        )
    )

    assert plan.heading_error_rad == 0.0
    assert plan.lateral_error_m == -1.0


def test_planner_does_not_turn_early_when_distant_lookahead_crosses_corner() -> None:
    planner = WaypointPathPlanner(
        route=[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)],
        lookahead_m=15.0,
    )

    plan = planner.plan(
        PerceptionFrame(
            ego=EgoState(position=(6.0, 0.0), heading_rad=0.0, speed_mps=0.0)
        )
    )

    assert plan.target == (20.0, 1.0)
    assert plan.heading_error_rad == 0.0


def test_planner_caps_speed_for_previewed_curve() -> None:
    planner = WaypointPathPlanner(
        route=[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0)],
        lookahead_m=5.0,
        curvature_preview_m=25.0,
        maximum_lateral_acceleration_mps2=2.0,
        minimum_curve_speed_mps=3.0,
    )

    plan = planner.plan(
        PerceptionFrame(
            ego=EgoState(position=(0.0, 0.0), heading_rad=0.0, speed_mps=0.0)
        )
    )

    assert plan.route_speed_cap_mps == pytest.approx(math.sqrt(2.0 / (math.pi / 40.0)))

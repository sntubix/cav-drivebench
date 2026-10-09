import pytest

from metadrive_starter.planning import PolylineFuturePath


def test_projects_an_object_around_a_corner_by_path_distance() -> None:
    path = PolylineFuturePath([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])

    projection = path.project((0.0, 0.0), (10.0, 8.0), max_distance_m=25.0)

    assert projection is not None
    assert projection.distance_along_path_m == pytest.approx(18.0)
    assert projection.cross_track_m == pytest.approx(0.0)
    assert projection.tangent_world == pytest.approx((0.0, 1.0))
    assert projection.ego_tangent_world == pytest.approx((1.0, 0.0))


def test_rejects_an_object_behind_the_ego() -> None:
    path = PolylineFuturePath([(0.0, 0.0), (10.0, 0.0)])

    assert path.project((5.0, 0.0), (2.0, 0.0), max_distance_m=20.0) is None


def test_respects_prediction_distance_along_the_route() -> None:
    path = PolylineFuturePath([(0.0, 0.0), (20.0, 0.0)])

    assert path.project((0.0, 0.0), (18.0, 0.0), max_distance_m=15.0) is None


def test_progress_never_jumps_back_to_an_earlier_segment() -> None:
    path = PolylineFuturePath([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

    path.project((10.0, 8.0), (5.0, 10.0), max_distance_m=20.0)
    projection = path.project((10.0, 7.0), (5.0, 10.0), max_distance_m=20.0)

    assert projection is not None
    assert projection.distance_along_path_m == pytest.approx(7.0)

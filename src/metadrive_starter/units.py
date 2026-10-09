from __future__ import annotations


KILOMETERS_PER_HOUR_PER_METER_PER_SECOND = 3.6


def mps_to_kmh(value: float) -> float:
    """Convert an internal speed for an explicitly km/h-labelled display."""
    return float(value) * KILOMETERS_PER_HOUR_PER_METER_PER_SECOND

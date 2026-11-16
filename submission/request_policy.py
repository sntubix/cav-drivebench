"""Assignment 3: when to ask the model.

DriveBench loads this file when you pass ``--submission`` to a run that asks
the model, for example:

    uv run metadrive-starter run --config configs/demo-vla-fixture.yaml --submission submission

Each run builds one ``RequestPolicy(settings)``, where ``settings`` holds the
instructor's request limits: ``request_budget``, the requests one episode may
attempt (None when unbounded); ``minimum_interval_s``; ``request_timeout_s``;
and ``action_horizon_s``, how long a command may hold authority. DriveBench
calls ``begin_episode()`` before an episode's first tick, then
``should_request(tick)`` on the control loop every control tick, before it
captures a camera frame. Return True to ask the model now, False to hold the
request back. ``tick`` carries:

- ``now_s``, the local ``scene`` measured now, ``ego_speed_mps``, and
  ``cruise_speed_mps``;
- ``active``, the assessment behind the model command that holds authority,
  or None while the local fallback drives, and ``issued_at_s``, when the model
  was asked for it;
- ``failure``, why a model request collected on this tick failed, if one did:
  ``timeout``, ``quota``, ``model_unavailable`` during an outage,
  ``invalid_response``, or ``transport``;
- ``in_flight``, True while an earlier request is unanswered;
- ``last_requested_at_s``, when this episode's latest request was sent;
- ``requests_remaining``, what is left of the budget, counting a request in
  flight as spent;
- ``route_speed_cap_mps``, the speed limit for the road curvature ahead, from
  the previous tick's plan, or None on a straight road.

Where these come from: ``RequestPolicySettings`` and ``RequestTick`` are
defined, with every field described, in src/metadrive_starter/vla/request_policy.py.
``tick.scene`` is the ``LocalScene`` of ``metadrive_starter.perception.scene``,
the scene your arbiter receives: its ``objects``, each a ``TrackedObject``,
and which lanes beside the car are available. ``tick.failure`` is a
``ModelFailureCategory`` of ``metadrive_starter.vla.provider``, and equals
the plain string, so ``tick.failure == "quota"`` holds. Import a type from its
module when you want to name it.

You may only ask less. True is a request, not a guarantee: DriveBench still
refuses while a request is in flight, sooner than ``settings.minimum_interval_s``
after the last one, and once the budget is spent. Every request the provider
receives counts against the budget, whether it succeeds, fails, times out, or
meets a quota or an outage. A ``should_request`` that raises or returns
anything but True or False holds that tick's request back, and the "monotone
restriction preserved" gate fails. It must return within one control step,
0.1 s at the default timing; the "control loop never blocked" gate times every
call.

Once the budget is spent, the model is never asked again that episode: the
last command keeps its authority to its horizon, then the local fallback drives
at the cruise speed with the safety floor and emergency braking.

As shipped, this asks every ``minimum_interval_s``, which is how DriveBench
asked before the request policy: it spends a long route's budget early and
asks on through an outage. Every place that needs your work is marked TODO.

The minimum interval can be raised, never lowered, in agent.yaml without
touching this file:

    vla:
      minimum_interval_s: 1.0   # may only be raised
"""

from __future__ import annotations

from metadrive_starter.vla.request_policy import RequestPolicySettings, RequestTick



class RequestPolicy:
    """Asks every ``minimum_interval_s``."""

    def __init__(self, settings: RequestPolicySettings) -> None:
        self.settings = settings
        # TODO: keep what the last request saw, so a change in the scene can be told.

    def begin_episode(self) -> None:
        # TODO: clear the history that __init__ keeps.
        pass

    def should_request(self, tick: RequestTick) -> bool:
        # TODO: ask when the scene changes, not on a timer: a new object in the
        # ego path, a lane beside the car becoming available, a lead vehicle
        # closing fast, or a curve ahead (tick.route_speed_cap_mps).
        # TODO: hold while the command that has authority still holds and
        # nothing changed (tick.active, tick.issued_at_s, settings.action_horizon_s).
        # TODO: back off after a failure, and longer on quota or
        # model_unavailable: asking through an outage spends the budget on failures.
        # TODO: keep a reserve of the budget (tick.requests_remaining) for what
        # the road has not shown yet.
        if tick.last_requested_at_s is None:
            return True
        return tick.now_s - tick.last_requested_at_s >= self.settings.minimum_interval_s

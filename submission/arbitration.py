"""Assignment 2: how far to trust the model.

DriveBench loads this file when you pass ``--submission`` to a run that asks
the model, for example:

    uv run metadrive-starter run --config configs/demo-vla-fixture.yaml --submission submission

Each run builds one ``Arbiter(settings)``, where ``settings`` holds the safety
floor's ``command_validation`` thresholds. DriveBench calls ``begin_episode()``
before an episode's first assessment, then two methods on the control loop:

- ``arbitrate(request)`` each time the model's assessment arrives. ``request``
  carries the assessment, the local scene measured now, the time now, when the
  model was asked, and the cruise speed. What it returns is what the safety
  floor validates.
- ``review(request)`` every control tick, after any new assessment has been
  arbitrated. ``request.active`` is the assessment behind the model command
  that holds authority, or None while the local fallback drives, and
  ``request.failure`` says why a model request collected on this tick failed,
  if one did. Return True to keep that command, False to end it now.

You may only take authority away. In ``arbitrate``:

- lower ``proposed_target_speed_mps``;
- lower ``confidence``: below ``settings.minimum_confidence`` the floor falls
  back to local driving;
- decline, by replacing ``proposed_action`` with ``REQUEST_FALLBACK``: the
  vehicle drops the model's command and drives on the local fallback until the
  next assessment you let through.

Only those three fields reach the vehicle. Return anything with more authority,
another action, or something other than a ``VLAAssessment``, or raise, and
DriveBench declines the assessment for you. A ``review`` that raises or returns
anything but True or False ends the command. Either way the "monotone
restriction preserved" gate fails. Both methods must return within one control
step, 0.1 s at the default timing; the "control loop never blocked" gate times
every call.

The local fallback drives the route at the cruise speed, with the safety floor
and emergency braking. Ending a model command takes away the model's authority,
not the vehicle's speed.

As shipped, this endorses every assessment and keeps every command to its
horizon, which is how DriveBench drove before arbitration. Each place to change
is marked TODO. Places to start:

- Hazard agreement carries signal. Compare the hazards the model reports with
  ``request.scene``: ``metadrive_starter.hazard_agreement.score_hazards`` scores
  them as the course measured them, and an assessment that invents a hazard or
  leaves out one in the ego path deserves less trust.
- Agreement is not enough. A model can report the stopped vehicle ahead and
  still propose ``KEEP_LANE`` at cruise speed; check the fields against each
  other too.
- Confidence has carried no measurable signal in the models measured so far.
  Test that claim before you rely on it.
- Assignment 2 asks for ``arbitrate`` only: leave ``review`` returning True.
  The runtime already ends each command at its horizon, aborts a lane change
  into a lane that is no longer clear, and brakes for what is ahead, and ending
  a command early hands control to the fallback at the cruise speed.
- Keep history across assessments if you filter over time or hold an action
  against flapping, and clear it in ``begin_episode``.

The floor's thresholds can be tightened, never loosened, in agent.yaml without
touching this file:

    command_validation:
      minimum_confidence: 0.7      # may only be raised
      maximum_target_speed_mps: 5   # may only be lowered
"""

from __future__ import annotations

from metadrive_starter.safety import CommandValidationSettings
from metadrive_starter.vla.arbitration import ArbitrationRequest, ReviewRequest
from metadrive_starter.vla.assessment import VLAAssessment


class Arbiter:
    """Endorses every assessment and keeps every command to its horizon."""

    def __init__(self, settings: CommandValidationSettings) -> None:
        self.settings = settings
        # TODO: keep any history your arbiter needs across assessments.

    def begin_episode(self) -> None:
        # TODO: clear any history that __init__ keeps.
        pass

    def arbitrate(self, request: ArbitrationRequest) -> VLAAssessment:
        # TODO: compare the hazards the assessment reports with request.scene, and
        # its action and speed with those hazards. Lower the speed or the
        # confidence, or decline, when you trust it less.
        return request.assessment

    def review(self, request: ReviewRequest) -> bool:
        return True

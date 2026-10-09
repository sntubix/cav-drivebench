# Assignment 1 rubric

[← Assignment 1 brief](README.md)

Assignment 1 carries 15% of the assignment marks. The assignments together make
up 60% of the project grade, and the oral defence at the end of the project the
other 40%. Assignment 1 is marked out of 100, automatically, from the hidden gate
run. There is no written report: your team explains and analyses its work at the
oral defence.

The course calendar gives the deadline. Late submissions are not accepted.

Once Assignment 1 is released, this rubric does not change. If a correction is
ever needed, it is announced and applies to every team alike.

## The floor: hidden structural gates

Grading runs the structural gates on your `submission/` against the hidden
routes, with the same command and code you run on the public ones.

- If every gate passes, or reports "not run", the assignment is marked from the
  hidden score.
- If any gate fails, the assignment earns nothing.

A hidden route that crashes, leaves the road, or does not arrive fails no gate:
it earns no driving credit, and the rest is marked as usual. You cannot drive
the hidden routes, so a tuning choice that leaves one of them costs that route,
not the assignment.

The gates you run are the gates that grade you, so a hidden gate that fails
should be a check you skipped. There are no resubmissions: the deadlines leave
time to run every check before handing in, and late submissions are not
accepted. A failure caused on the grading side is graded again by staff.

## The mark

The mark is the score on the hidden routes, defined in
[The score](README.md#the-score): half from the six implementation checks, half
from the mean driving credit over the hidden routes. A score of 90 earns 90
marks.

The implementation checks test your controller's implementation with gains of
their own. Driving credit comes from how your controller, with your
`agent.yaml`, drives the hidden routes. A complete PID on the starting gains in
[`configs/pid.yaml`](../../configs/pid.yaml) earns about 63 of 100 on the hidden routes, and the instructor
reference's tuned gains about 85; closer speed and lane tracking earn the rest,
so tuning counts.

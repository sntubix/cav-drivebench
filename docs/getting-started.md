# Getting started

[← DriveBench README](../README.md) · [Next: Assignment 2 brief →](../course/assignment-2/README.md)

The basics: get the code, install it, check that it runs, and know where your
work goes, how to check it, and how to hand it in. For rendered
demonstrations, Docker, platforms, and configuration, see
[Running and configuring DriveBench](running-and-configuring.md). Run all
commands from the repository root unless stated otherwise.

## Get the code and stay up to date

The course publishes DriveBench as a git repository, and every assignment
arrives as a new release on it. All you need is a clone of it that pulls each
release. The instructors never look at your repository: you hand in a zip on
Moodle, as [Hand in](#hand-in) describes.

### Recommended: a team repository

For working together, we recommend a private repository for your team that
pulls each release from the course repository, kept as a remote called
`upstream`. Your work lives in your repository; the course repository only ever
brings updates.

To set it up, one team member creates an empty private repository for the
team on your git host, with no README, licence, or other files, then:

```bash
git clone https://github.com/sntubix/cav-drivebench.git drivebench
cd drivebench
git remote rename origin upstream
git remote add origin <your team's repository URL>
git push -u origin main
```

Everyone else on the team clones the team's repository and adds the same
`upstream`:

```bash
git clone <your team's repository URL> drivebench
cd drivebench
git remote add upstream https://github.com/sntubix/cav-drivebench.git
```

Commit your work and push it to `origin`, your team's repository, as usual.
Never push to `upstream`: it belongs to the course.

Set up your repository as above rather than with your git host's "fork" or
"use this template". A template copy shares no history with the course
repository, so releases no longer merge into it, and a fork of a public
repository is public too.

### Without a team repository

Clone the course repository and work in that clone:

```bash
git clone https://github.com/sntubix/cav-drivebench.git drivebench
cd drivebench
```

The course repository is then `origin`: wherever [Pull a release](#pull-a-release)
says `upstream`, write `origin`, and skip `git push`.

### Pull a release

When the course announces a release, pull it into your work, then push the
result to your team's repository:

```bash
git pull --no-rebase --no-edit upstream main
git push
```

A release changes only files outside `submission/`, and adds new stubs inside it
as new files, such as the next assignment's. It never changes a file in
`submission/`, or a course notebook, that has already shipped. Every assignment
asks you to change only `submission/`, and the notebooks are yours to edit, so
the pull merges without a conflict. If git reports one anyway, it is in another
file you changed: take the course's version of that file, then finish the merge:

```bash
git checkout upstream/main -- <file>
git commit --no-edit
```

## Install

DriveBench supports Python `>=3.10,<3.12` because MetaDrive `0.4.3` does not
support Python 3.12. Install [`uv`](https://docs.astral.sh/uv/) if it is not
already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment, including the development checks:

```bash
uv sync --group dev
```

## Check that it works

First check the wiring without launching MetaDrive:

```bash
uv run metadrive-starter smoke --submission submission
```

Then drive one route headless with the autopilot profile. The first launch
downloads MetaDrive's assets, which takes a minute or more:

```bash
uv run metadrive-starter run \
  --config configs/demo-autopilot.yaml \
  --submission submission \
  --headless \
  --steps 1000
```

The summary should report `arrived: true`, `crashed: false`, and
`went_off_road: false`.

## Your submission

Your team's work lives only in `submission/`: never edit a file outside it.
Each assignment's brief says which files to write there and which keys
`agent.yaml` accepts; for Assignment 1, see
[The task](../course/assignment-1/README.md#the-task) and
[Tuning](../course/assignment-1/README.md#tuning).

DriveBench drives with the code in the folder you pass with `--submission`.
Without it, `run` and `smoke` stop and ask for one:

```bash
uv run metadrive-starter smoke --submission submission
uv run metadrive-starter run --submission submission --headless --map C
```

`smoke` loads your code and computes one control tick without launching
MetaDrive, so run it first after every change. The summary's `controller` field
names what drove: `submission.controller.Controller` is yours. A file that
cannot be loaded stops the run before MetaDrive starts, with a message naming
what was expected. `run` drives one route and prints a JSON summary.

[`submission/agent.yaml`](../submission/agent.yaml) is an overlay on the instructor configuration, not a full
configuration: write only the keys you change. A key your assignment does not
permit, or a value outside its bounds, is rejected by name. Print the
configuration your runs will use:

```bash
uv run metadrive-starter config --submission submission
```

### Check your work

The structural gates check your submission the way grading does. Each
assignment has its own gates, so run the `gates` command its brief gives; the
bare command runs Assignment 1's, which never load an `observation.py` or an
`arbitration.py`. For Assignment 2:

```bash
uv run metadrive-starter gates --manifest configs/gates-assignment-2.yaml --config configs/demo-vla-fixture.yaml
```

`gates` checks `submission/` unless you pass `--submission` with another
folder. It lists each gate, then `PASSED` or `FAILED`. Each gate reports on its
own, so one broken piece never hides another. Grading runs the same gates on hidden routes, so a gate that
fails there is one you could have run yourself. Each brief lists its gates and,
if the assignment has one, how its score is computed. While iterating,
`--scenario straight` drives just that route (repeat it to pick several) and
marks the report partial; before handing in, run them all.

## Hand in

Every assignment is handed in the same way: one zip of your `submission/`
folder, uploaded to Moodle by the assignment's deadline. One member uploads it
for the whole team. Each brief lists the files the zip must hold.

Make the zip from the repository root, so that it holds the `submission/`
folder itself:

```bash
mkdir -p tmp
uv run python -m zipfile -c tmp/submission.zip submission
```

Before uploading, check the zip: unpack it into a new folder, and run the
assignment's gates on what came out by adding `--submission` with that folder
to the `gates` command its brief gives. The report should match the one for
`submission/`. For Assignment 2:

```bash
check=$(mktemp -d tmp/zip-check-XXXXXX)
uv run python -m zipfile -e tmp/submission.zip "$check"
uv run metadrive-starter gates --manifest configs/gates-assignment-2.yaml --config configs/demo-vla-fixture.yaml --submission "$check/submission"
```

The first line makes a new, empty folder for every check, so files left from an
earlier check never stand in for one the zip lacks.

You may replace your upload as often as you like until the deadline. Only the
last upload is graded, after the deadline closes, and nothing is accepted after
it.

The zip is all the instructors receive: they never look at your repository.
Grading places your `submission/` in a clean copy of the course repository, so
a change to any file outside it never reaches the grader. Never edit one: the
structural gates flag it before you hand in.

## Where to go next

Read the briefs, in order; each builds on the one before:
[Assignment 1](../course/assignment-1/README.md), then
[Assignment 2](../course/assignment-2/README.md). For demonstrations,
Docker, platforms, and configuration, see
[Running and configuring DriveBench](running-and-configuring.md).

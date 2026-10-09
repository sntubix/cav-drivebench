"""Build the two releases of one commit: the student and instructor archives.

The student release ships the structural-gate runner, the public scenarios, and
the assignment packs. The instructor release is the whole commit, and adds what
course/instructor/ holds: hidden manifests and tests, reference solutions, and
grading integration. Everything withheld from teams lives there, because
repository location is not access control: only this module's exclusions keep it
out of the student release. The student release also leaves out maintainer
documentation: the guides and notes at the repository root, and docs/adr/ and
docs/agents/, and every path course/instructor/release-plan.yaml holds back
until its assignment is released. Before writing either release the build looks for withheld
material that escaped, for a student document linking to a withheld file, for a
shipped stub that its marked reference no longer generates, and for credentials
in either release, and refuses to write anything if it finds one.

Teams clone the student repository and pull each release into their work, so
``--student-repo`` commits a release onto a clone of it, as one more commit on
its history. It refuses a release that would change or remove a file under
submission/ that an earlier release shipped, since teams own those files.

Run from the repository root:

    uv run python -m metadrive_starter.release
    uv run python -m metadrive_starter.release --student-repo ../cav-drivebench
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Mapping

import yaml

from metadrive_starter.gates import write_release_manifest
from metadrive_starter.submission import PROJECT_ROOT

# Every directory a student release leaves out. Put withheld material, such as
# hidden manifests and tests, fault schedules, reference solutions, tuned
# profiles, and grading integration, under course/instructor/ rather than adding
# a path here. The other two hold maintainer documentation.
STUDENT_EXCLUSIONS = (
    PurePosixPath("course/instructor"),
    PurePosixPath("docs/adr"),
    PurePosixPath("docs/agents"),
)
# The only documents at the repository root a student release ships. Every other
# root document, such as AGENTS.md, CLAUDE.md, CONTRIBUTING.md, and the
# maintainers' notes, is for maintainers, so a new one stays out of student
# releases unless it is listed here.
STUDENT_ROOT_DOCUMENTS = frozenset({"README.md", "CONTEXT.md"})
# Paths held out of student releases until their assignment is released. Unlike
# withheld material they are not secret, only not shipped yet.
RELEASE_PLAN = PurePosixPath("course/instructor/release-plan.yaml")
_DOCUMENT_SUFFIXES = frozenset({".md", ".pdf"})

# Credentials resolve from the environment, so no shipped file ever holds one.
_SECRETS = (
    ("a private key", re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    (
        "a Google credential file",
        re.compile(rb'"type"\s*:\s*"(?:service_account|authorized_user|external_account)"'),
    ),
    ("a Google API key", re.compile(rb"AIza[0-9A-Za-z_-]{35}")),
    ("a Google OAuth token", re.compile(rb"ya29\.[0-9A-Za-z_-]{20,}")),
    ("a Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{30,}")),
    ("a GitHub token", re.compile(rb"gh[oprsu]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}")),
)
_CREDENTIAL_FILE = re.compile(
    r"\.env(?:\.(?!example$).+)?|.+\.(?:pem|key|p12|pfx)|application_default_credentials\.json"
)
# A Markdown link's target: [text](target) or [text](<target> "title").
_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


class ReleaseError(Exception):
    """The releases cannot be built, or would ship what they must not."""


@dataclass(frozen=True)
class ReleaseArchives:
    """The student and instructor archives of one commit."""

    student: Path
    instructor: Path


def build_releases(
    repository: Path | str,
    output_dir: Path | str,
    *,
    revision: str = "HEAD",
) -> ReleaseArchives:
    """Write ``drivebench-{student,instructor}-<commit>.tar.gz`` from one commit.

    Only committed files ship. Each archive unpacks into one directory named
    like it, carries ``release-manifest.json``, and is byte-for-byte the same
    whenever the same commit is released. Both are written, or neither: an
    existing archive is never overwritten.
    """
    repository = Path(repository)
    commit = _git(
        repository, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"
    ).decode().strip()
    committed_at = int(_git(repository, "show", "-s", "--format=%ct", commit))
    archives = ReleaseArchives(
        student=Path(output_dir) / f"drivebench-student-{commit[:12]}.tar.gz",
        instructor=Path(output_dir) / f"drivebench-instructor-{commit[:12]}.tar.gz",
    )
    for archive in (archives.student, archives.instructor):
        if archive.exists():
            raise ReleaseError(f"{archive} already exists; a release is never overwritten")
    exported = _git(repository, "archive", "--format=tar", commit)
    with tempfile.TemporaryDirectory() as staging:
        student = Path(staging) / _directory(archives.student)
        instructor = Path(staging) / _directory(archives.instructor)
        _stage_checked(exported, student, instructor)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _write_archive(student, archives.student, committed_at)
        try:
            _write_archive(instructor, archives.instructor, committed_at)
        except BaseException:
            archives.student.unlink()
            raise
    return archives


def publish_student_release(
    repository: Path | str,
    student_repository: Path | str,
    *,
    revision: str = "HEAD",
) -> str | None:
    """Commit the student release of one commit onto a clone of the student repository.

    Each release lands as one more commit on the student repository's history,
    never as a new history, so a team that cloned it can pull every release into
    its own work. Returns the new commit, or None when the clone already holds
    this release. Pushing is left to the caller.
    """
    commit = stage_student_release(repository, student_repository, revision=revision)
    if commit is None:
        return None
    target = Path(student_repository)
    _git(target, "commit", "-q", "-m", release_message(commit))
    return _git(target, "rev-parse", "HEAD").decode().strip()


def release_message(commit: str) -> str:
    """The commit message of the student release built from ``commit``."""
    return f"DriveBench release {commit[:12]}\n\nBuilt from {commit} of the course repository."


def stage_student_release(
    repository: Path | str,
    student_repository: Path | str,
    *,
    revision: str = "HEAD",
) -> str | None:
    """Write and stage the student release of one commit in a clone of the student repository.

    A release that would change or remove a file under submission/, or a course
    notebook, that the clone already holds is refused: teams edit those files,
    and a change to one would collide with their work when they pull. Returns the course commit
    released, or None when the clone already holds this release. Committing is
    left to the caller.
    """
    repository = Path(repository)
    target = Path(student_repository)
    commit = _git(
        repository, "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"
    ).decode().strip()
    if not (target / ".git").exists():
        raise ReleaseError(
            f"{target} is not a git repository; clone the student repository first"
        )
    if _git(target, "status", "--porcelain").strip():
        raise ReleaseError(f"{target} has uncommitted changes; release into a clean clone")
    exported = _git(repository, "archive", "--format=tar", commit)
    shipped = [
        name for name in _git(target, "ls-files", "-z").decode().split("\0") if name
    ]
    with tempfile.TemporaryDirectory() as staging:
        student = Path(staging) / "student"
        _stage_checked(exported, student, Path(staging) / "instructor")
        problems = [
            f"{name} would {'change' if (student / name).is_file() else 'disappear'}"
            for name in shipped
            if _team_owned(PurePosixPath(name))
            and not (
                (student / name).is_file()
                and (student / name).read_bytes() == (target / name).read_bytes()
            )
        ]
        if problems:
            raise ReleaseError(
                "refusing to release: teams own every file under submission/, and every "
                "course notebook, that has shipped, so a release adds such files but "
                "never changes one:\n"
                + "\n".join(f"  {problem}" for problem in problems)
            )
        for name in shipped:
            (target / name).unlink()
        _remove_empty_directories(target)
        for path, name, _ in _files(student):
            destination = target.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    _git(target, "add", "-A")
    if not _git(target, "status", "--porcelain").strip():
        return None
    return commit


def _team_owned(path: PurePosixPath) -> bool:
    """Whether teams edit ``path`` once it ships: their submission, and the course
    notebooks, which hold copies of it to experiment on."""
    return path.parts[0] == "submission" or (path.parts[0] == "course" and path.suffix == ".ipynb")


def withheld_from_students(path: PurePosixPath) -> bool:
    """Whether the student release leaves out ``path``, a path in the repository."""
    if any(path == excluded or excluded in path.parents for excluded in STUDENT_EXCLUSIONS):
        return True
    return (
        len(path.parts) == 1
        and path.suffix.lower() in _DOCUMENT_SUFFIXES
        and path.name not in STUDENT_ROOT_DOCUMENTS
    )


def _stage_checked(exported: bytes, student: Path, instructor: Path) -> None:
    """Stage both trees with their manifests, or refuse what the student one must not hold."""
    _stage(exported, instructor, lambda path: False)
    held = _held_paths(instructor)
    withheld = _stage(exported, student, withheld_from_students, hold=_under(held))
    write_release_manifest(student)
    write_release_manifest(instructor)
    # The instructor tree holds every file the student tree does.
    problems = [
        *_withheld_leaks(student, withheld),
        *_links_to_withheld(student, instructor),
        *_stale_stubs(student, instructor, held),
        *_credentials(instructor),
    ]
    if problems:
        raise ReleaseError(
            "refusing to release:\n" + "\n".join(f"  {problem}" for problem in problems)
        )


def _held_paths(instructor: Path) -> tuple[PurePosixPath, ...]:
    """The paths the release plan holds back, each checked to name a path in the commit."""
    plan = instructor.joinpath(*RELEASE_PLAN.parts)
    if not plan.is_file():
        return ()
    data = yaml.safe_load(plan.read_text(encoding="utf-8"))
    entries = data.get("held") if isinstance(data, dict) else None
    if not isinstance(data, dict) or set(data) != {"held"} or not isinstance(entries, list):
        raise ReleaseError(f"{RELEASE_PLAN} must hold one list, held")
    held = []
    for entry in entries:
        path = PurePosixPath(entry) if isinstance(entry, str) else None
        if path is None or path.is_absolute() or ".." in path.parts or not path.parts:
            raise ReleaseError(f"{RELEASE_PLAN}: {entry!r} is not a path in the repository")
        if not instructor.joinpath(*path.parts).exists():
            raise ReleaseError(f"{RELEASE_PLAN}: {entry} is not in the released commit")
        held.append(path)
    return tuple(held)


def _under(paths: tuple[PurePosixPath, ...]) -> Callable[[PurePosixPath], bool]:
    return lambda path: any(path == held or held in path.parents for held in paths)


def _remove_empty_directories(tree: Path) -> None:
    for directory, _, _ in sorted(os.walk(tree), key=lambda entry: -len(entry[0])):
        path = Path(directory)
        if path != tree and ".git" not in path.relative_to(tree).parts and not any(path.iterdir()):
            path.rmdir()


def _directory(archive: Path) -> str:
    return archive.name.removesuffix(".tar.gz")


def _git(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ReleaseError("git is required to export a commit") from exc
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise ReleaseError(f"git {arguments[0]} failed: {detail or 'no such commit'}")
    return result.stdout


def _stage(
    exported: bytes,
    tree: Path,
    withhold: Callable[[PurePosixPath], bool],
    *,
    hold: Callable[[PurePosixPath], bool] = lambda path: False,
) -> dict[str, str]:
    """Write a git export under ``tree``, less every path ``withhold`` or ``hold`` names.

    Returns the excluded files, keyed by SHA-256, so that a copy of one outside
    the excluded paths can be recognised.
    """
    withheld: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(exported)) as export:
        for member in export:
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ReleaseError(f"{member.name} is not a regular file in the repository")
            source = export.extractfile(member)
            assert source is not None
            data = source.read()
            if withhold(path):
                withheld[_sha256(data)] = member.name
                continue
            if hold(path):
                continue
            target = tree.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    return withheld


def _withheld_leaks(student: Path, withheld: Mapping[str, str]) -> Iterator[str]:
    """Name every file of the student tree that carries withheld material."""
    for path, name, data in _files(student):
        digest = _sha256(data)
        if withheld_from_students(PurePosixPath(name)):
            yield f"{name} is under an excluded path"
        if data and digest in withheld:
            yield f"{name} is a copy of {withheld[digest]}"
        if _is_hidden_manifest(path, data):
            yield f"{name} is a hidden manifest"


def _links_to_withheld(student: Path, instructor: Path) -> Iterator[str]:
    """Name every student document that links to a file only the instructor tree holds."""
    for path, name, data in _files(student):
        if path.suffix != ".md":
            continue
        for match in _LINK.finditer(data.decode("utf-8", errors="replace")):
            target = match.group(1).split("#", 1)[0]
            if not target or _URL_SCHEME.match(target) or target.startswith("/"):
                continue
            linked = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
            if linked.startswith("../"):
                continue
            if not (student / linked).exists() and (instructor / linked).exists():
                yield f"{name} links to {linked}, which the student release leaves out"


def _stale_stubs(
    student: Path, instructor: Path, held: tuple[PurePosixPath, ...] = ()
) -> Iterator[str]:
    """Name every shipped stub that its marked reference does not generate.

    Runs the released commit's own course/instructor/make_stubs.py, so the
    check always matches the markers of the references it reads.
    """
    script = instructor / "course" / "instructor" / "make_stubs.py"
    if not script.is_file():
        return
    spec = importlib.util.spec_from_file_location("_released_make_stubs", script)
    assert spec is not None and spec.loader is not None
    stubs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stubs)
    references = instructor / "course" / "instructor" / "reference-submission"
    try:
        for reference in stubs.marked_references(references):
            name = f"submission/{reference.name}"
            if _under(held)(PurePosixPath(name)):
                continue
            expected = stubs.make_stub(reference.read_text(encoding="utf-8"), name=reference.name)
            shipped = student / "submission" / reference.name
            if not shipped.is_file() or shipped.read_text(encoding="utf-8") != expected:
                yield (
                    f"{name} is not what its marked reference generates; run "
                    "course/instructor/make_stubs.py"
                )
    except stubs.StubError as exc:
        yield f"a marked reference cannot generate its stub: {exc}"


def _credentials(tree: Path) -> Iterator[str]:
    """Name every file of ``tree`` that is or holds a credential."""
    for path, name, data in _files(tree):
        if _CREDENTIAL_FILE.fullmatch(path.name):
            yield f"{name} is a credential file"
        for secret, pattern in _SECRETS:
            if pattern.search(data):
                yield f"{name} contains {secret}"


def _files(tree: Path) -> Iterator[tuple[Path, str, bytes]]:
    for path in sorted(path for path in tree.rglob("*") if path.is_file()):
        yield path, path.relative_to(tree).as_posix(), path.read_bytes()


def _is_hidden_manifest(path: Path, data: bytes) -> bool:
    try:
        if path.suffix in {".yaml", ".yml"}:
            document = yaml.safe_load(data)
        elif path.suffix == ".json":
            document = json.loads(data)
        else:
            return False
    except (yaml.YAMLError, ValueError):
        return False
    return isinstance(document, dict) and document.get("visibility") == "hidden"


def _write_archive(tree: Path, archive: Path, mtime: int) -> None:
    # Fixed order, owners, modes, and times, and no gzip timestamp or name, so
    # the same commit always produces the same bytes.
    with archive.open("xb") as raw:
        try:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                    for path in [tree, *sorted(tree.rglob("*"))]:
                        _add(bundle, path, path.relative_to(tree.parent).as_posix(), mtime)
        except BaseException:
            # A partial archive must never pass for a release.
            archive.unlink()
            raise


def _add(bundle: tarfile.TarFile, path: Path, name: str, mtime: int) -> None:
    info = tarfile.TarInfo(name)
    info.mtime = mtime
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        bundle.addfile(info)
        return
    status = path.stat()
    info.mode = 0o755 if status.st_mode & 0o111 else 0o644
    info.size = status.st_size
    with path.open("rb") as content:
        bundle.addfile(info, content)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m metadrive_starter.release",
        description=(
            "Build the student and instructor release archives from one commit, "
            "and print their paths, student first. The student archive leaves out "
            + ", ".join(f"{path}/" for path in STUDENT_EXCLUSIONS)
            + ", and every root document but "
            + " and ".join(sorted(STUDENT_ROOT_DOCUMENTS))
            + "."
        ),
    )
    parser.add_argument(
        "--revision",
        default="HEAD",
        help="commit to release; uncommitted changes never ship",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/releases"),
        help="directory the archives are written to",
    )
    parser.add_argument(
        "--student-repo",
        type=Path,
        help=(
            "a clone of the student repository: commit the student release onto its "
            "history instead of writing archives; pushing is left to you"
        ),
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="with --student-repo: stage the release in the clone, and leave committing to you",
    )
    args = parser.parse_args(argv)
    if args.no_commit and args.student_repo is None:
        parser.error("--no-commit needs --student-repo")
    try:
        if args.student_repo is not None and args.no_commit:
            staged = stage_student_release(
                PROJECT_ROOT, args.student_repo, revision=args.revision
            )
            if staged is None:
                print(f"{args.student_repo} already holds this release")
            else:
                print(
                    f"staged the release of {staged[:12]} in {args.student_repo}; review "
                    f"it, then commit it with this message and push:\n\n"
                    + release_message(staged)
                )
            return 0
        if args.student_repo is not None:
            published = publish_student_release(
                PROJECT_ROOT, args.student_repo, revision=args.revision
            )
            if published is None:
                print(f"{args.student_repo} already holds this release")
            else:
                print(f"committed {published} to {args.student_repo}; review it, then push")
            return 0
        archives = build_releases(PROJECT_ROOT, args.output_dir, revision=args.revision)
    except ReleaseError as exc:
        parser.exit(1, f"release: {exc}\n")
    print(archives.student)
    print(archives.instructor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

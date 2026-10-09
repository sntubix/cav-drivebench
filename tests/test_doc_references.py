"""Every file reference in the documentation points at something that exists."""

import posixpath
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIPPED = {".git", ".claude", ".venv", "tmp", "artifacts", "test_artifacts", "node_modules"}
# A repository path written as code: the first part is a top-level directory.
REPOSITORY_DIRECTORIES = {"configs", "course", "docs", "fixtures", "scripts", "src", "submission", "tests"}
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
CODE_PATH = re.compile(r"(?<!\[)`([^`\s<>*{}]+/[^`\s<>*{}]*)`(?!\]\()")
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def _documents() -> list[Path]:
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*.md")
        if not SKIPPED.intersection(path.relative_to(PROJECT_ROOT).parts)
    )


def _prose(path: Path) -> list[str]:
    """The document's lines outside fenced code blocks."""
    lines, fenced = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            lines.append(line)
    return lines


def _anchors(path: Path) -> set[str]:
    """GitHub's heading anchors for ``path``."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in _prose(path):
        heading = re.match(r"#{1,6}\s+(.*)", line)
        if heading:
            slug = re.sub(r"[^\w\- ]", "", re.sub(r"[`*_]", "", heading.group(1).strip().lower()))
            slug = slug.replace(" ", "-")
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _name(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


DOCUMENTS = _documents()


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_every_relative_link_reaches_an_existing_file_and_heading(document: Path) -> None:
    problems = []
    for line in _prose(document):
        for match in LINK.finditer(line):
            target, _, anchor = match.group(1).partition("#")
            if URL.match(target):
                continue
            linked = (document.parent / target).resolve() if target else document
            if not linked.exists():
                problems.append(f"{match.group(1)}: no such file")
            elif anchor and linked.suffix == ".md" and anchor not in _anchors(linked):
                problems.append(f"{match.group(1)}: no such heading")

    assert problems == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=_name)
def test_every_repository_path_names_an_existing_file_and_links_it_below_the_root(
    document: Path,
) -> None:
    # Paths are written from the repository root. Below the root, a bare path
    # would resolve against the document's own folder, so it must be a link.
    problems = []
    for line in _prose(document):
        for match in CODE_PATH.finditer(line):
            path = match.group(1).rstrip("/")
            if path.split("/")[0] not in REPOSITORY_DIRECTORIES:
                continue
            if not (PROJECT_ROOT / path).exists():
                problems.append(f"{path}: no such file or directory")
            elif document.parent != PROJECT_ROOT and (PROJECT_ROOT / path).is_file():
                relative = posixpath.relpath(path, _name(document.parent))
                problems.append(f"{path}: write it as [`{path}`]({relative})")

    assert problems == []

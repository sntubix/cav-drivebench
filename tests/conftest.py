import os
from typing import Iterator

import pytest

from metadrive_starter.submission import INSTRUCTOR_DIR, PROJECT_ROOT


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Student releases leave out course/instructor/, including the reference
    # controller and hidden material, and hold other paths back until their
    # assignment is released, so tests that need any of them skip there.
    instructor = pytest.mark.skip(reason="needs course/instructor/, which this release leaves out")
    for item in items:
        if item.get_closest_marker("instructor") is not None and not INSTRUCTOR_DIR.is_dir():
            item.add_marker(instructor)
        for marker in item.iter_markers("needs"):
            missing = [path for path in marker.args if not (PROJECT_ROOT / path).exists()]
            if missing:
                item.add_marker(
                    pytest.mark.skip(reason=f"needs {', '.join(missing)}, which this release holds back")
                )


@pytest.fixture(autouse=True)
def _restore_environment() -> Iterator[None]:
    # MetaDrive's asset loader sets PYTHONUTF8=on for the whole process, a value
    # Python rejects at startup, so any later test that starts a Python
    # subprocess with a copy of os.environ would fail.
    saved = dict(os.environ)
    yield
    if dict(os.environ) != saved:
        os.environ.clear()
        os.environ.update(saved)

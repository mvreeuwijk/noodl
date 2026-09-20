from pathlib import Path

import noodl


def test_version():
    assert noodl.__version__


def test_the_imported_package_is_this_checkouts_src():
    """A worktree shares the venv with the main checkout, whose editable install would
    otherwise win: every test must exercise the source tree it lives next to."""
    repo = Path(__file__).resolve().parents[1]
    assert Path(noodl.__file__).resolve().is_relative_to(repo / "src")

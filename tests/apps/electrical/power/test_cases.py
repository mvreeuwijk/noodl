"""Fetching the MATPOWER standard cases on first use (no network: urllib is replaced)."""

import hashlib
import io

import pytest

from noodl.apps.electrical.power import cases
from noodl.apps.electrical.power.cases import (
    MATPOWER_CASES,
    MatpowerCaseDigestError,
    MatpowerCaseUnavailableError,
    matpower_case,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


@pytest.fixture
def served(monkeypatch):
    """Serve `files[name]` for any URL ending in `/<name>.m`; record the URLs asked for."""
    files: dict[str, bytes] = {}
    asked: list[str] = []

    def urlopen(url, timeout):
        asked.append(url)
        return io.BytesIO(files[url.rsplit("/", 1)[1][:-2]])

    monkeypatch.setattr(cases.urllib.request, "urlopen", urlopen)
    return files, asked


@pytest.fixture
def table(monkeypatch):
    """Replace the known digests by those of small stand-in files."""
    known = dict(MATPOWER_CASES)
    monkeypatch.setattr(cases, "MATPOWER_CASES", known)
    return known


def test_downloads_verifies_and_caches(tmp_path, served, table):
    files, asked = served
    files["case14"] = b"function mpc = case14\n"
    table["case14"] = _digest(files["case14"])
    path = matpower_case("case14", cache_dir=tmp_path)
    assert path == tmp_path / "matpower" / "8.1" / "case14.m"
    assert path.read_bytes() == files["case14"]
    assert asked == [f"https://raw.githubusercontent.com/MATPOWER/matpower/"
                     f"{cases.MATPOWER_COMMIT}/data/case14.m"]
    assert matpower_case("case14", cache_dir=tmp_path) == path
    assert len(asked) == 1, "a verified cached file is not downloaded again"


def test_cache_dir_from_environment(tmp_path, served, table, monkeypatch):
    files, _ = served
    files["case30"] = b"x\r\ny\r\n"
    table["case30"] = _digest(b"x\ny\n")
    monkeypatch.setenv("NOODL_CACHE_DIR", str(tmp_path))
    assert matpower_case("case30") == tmp_path / "matpower" / "8.1" / "case30.m"


def test_a_download_with_the_wrong_digest_is_refused_and_not_cached(tmp_path, served, table):
    files, _ = served
    files["case118"] = b"not the case\n"
    with pytest.raises(MatpowerCaseDigestError, match=r"case118.*SHA-256.*" + table["case118"]):
        matpower_case("case118", cache_dir=tmp_path)
    assert not (tmp_path / "matpower" / "8.1" / "case118.m").exists()


def test_a_cached_file_with_the_wrong_digest_is_refused(tmp_path, served, table):
    _, asked = served
    target = tmp_path / "matpower" / "8.1" / "case9.m"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"edited\n")
    with pytest.raises(MatpowerCaseDigestError, match=r"case9\.m.*delete it"):
        matpower_case("case9", cache_dir=tmp_path)
    assert asked == []


def test_a_manually_placed_file_is_used(tmp_path, served, table):
    _, asked = served
    target = tmp_path / "matpower" / "8.1" / "case9.m"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"placed by hand\r\n")
    table["case9"] = _digest(b"placed by hand\n")
    assert matpower_case("case9", cache_dir=tmp_path) == target and asked == []


def test_unknown_names_are_refused():
    with pytest.raises(ValueError, match=r"'case2383wp'.*case9, case14, case30, case118"):
        matpower_case("case2383wp")
    with pytest.raises(ValueError, match="'../case9'"):
        matpower_case("../case9")


def test_offline_names_the_file_to_place_by_hand(tmp_path, monkeypatch):
    # Port 9 (discard) on the loopback interface: the connection is refused at once.
    monkeypatch.setattr(cases, "MATPOWER_SOURCE", "http://127.0.0.1:9/{name}.m")
    target = tmp_path / "matpower" / "8.1" / "case9.m"
    with pytest.raises(MatpowerCaseUnavailableError) as info:
        matpower_case("case9", cache_dir=tmp_path)
    message = str(info.value)
    assert "http://127.0.0.1:9/case9.m" in message and str(target) in message
    assert MATPOWER_CASES["case9"] in message
    assert isinstance(info.value, OSError) and not target.exists()


def test_a_derived_case_reports_its_base_case_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, "MATPOWER_SOURCE", "http://127.0.0.1:9/{name}.m")
    with pytest.raises(MatpowerCaseUnavailableError, match=r"case9\.m"):
        matpower_case("case9_features", cache_dir=tmp_path)


@pytest.mark.external
def test_the_feature_case_is_derived_from_the_fetched_case9(tmp_path):
    try:
        path = matpower_case("case9_features", cache_dir=tmp_path)
    except MatpowerCaseUnavailableError as exc:
        pytest.skip(str(exc))
    assert path == tmp_path / "matpower" / "8.1" / "case9_features.m"
    assert (tmp_path / "matpower" / "8.1" / "case9.m").exists()
    assert _digest(path.read_bytes()) == MATPOWER_CASES["case9_features"]
    assert path.read_text(encoding="utf-8").startswith("function mpc = case9_features\n")

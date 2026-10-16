"""Fetching the MATPOWER standard cases on first use (no network: urllib is replaced)."""

import concurrent.futures
import hashlib
import http.client
import io
import threading
from pathlib import Path

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


def test_concurrent_fetchers_both_get_the_verified_file(tmp_path, served, table, monkeypatch):
    files, asked = served
    files["case14"] = b"function mpc = case14\n"
    table["case14"] = _digest(files["case14"])
    both_asking = threading.Barrier(2, timeout=10)
    serve = cases.urllib.request.urlopen

    def urlopen(url, timeout):
        both_asking.wait()          # neither writes before both have started downloading
        return serve(url, timeout)

    monkeypatch.setattr(cases.urllib.request, "urlopen", urlopen)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(lambda _: matpower_case("case14", cache_dir=tmp_path), range(2)))
    target = tmp_path / "matpower" / "8.1" / "case14.m"
    assert paths == [target, target] and len(asked) == 2
    assert target.read_bytes() == files["case14"]
    assert list(target.parent.glob("*.partial")) == []


def test_a_file_another_fetcher_holds_open_is_kept_when_it_verifies(tmp_path, served, table,
                                                                   monkeypatch):
    files, _ = served
    files["case30"] = b"function mpc = case30\n"
    table["case30"] = _digest(files["case30"])
    target = tmp_path / "matpower" / "8.1" / "case30.m"

    def replace_while_held(src, dst):   # Windows: the target is open in another process
        Path(dst).write_bytes(placed)
        raise PermissionError(13, "in use", str(dst))

    monkeypatch.setattr(cases.os, "replace", replace_while_held)
    placed = files["case30"]
    assert matpower_case("case30", cache_dir=tmp_path) == target
    assert list(target.parent.glob("*.partial")) == []
    target.unlink()
    placed = b"something else\n"
    monkeypatch.setattr(cases.time, "sleep", lambda s: None)
    with pytest.raises(MatpowerCaseUnavailableError, match="case30"):
        matpower_case("case30", cache_dir=tmp_path)
    assert list(target.parent.glob("*.partial")) == []


def test_a_target_mid_replace_is_retried_until_it_verifies(tmp_path, monkeypatch):
    data = b"function mpc = case9" + bytes([10])
    target = tmp_path / "case9.m"
    target.write_bytes(data)
    real_read, reads, sleeps = Path.read_bytes, [], []

    def read_bytes(self):
        reads.append(1)
        if len(reads) <= 2:             # the other fetcher's replace is still in progress
            raise PermissionError(13, "mid-replace", str(self))
        return real_read(self)

    def replace(src, dst):
        raise PermissionError(5, "held", str(dst))

    monkeypatch.setattr(cases.os, "replace", replace)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(cases.time, "sleep", sleeps.append)
    cases._write(target, data, _digest(data))
    assert len(reads) == 3 and len(sleeps) == 2
    assert real_read(target) == data
    assert list(tmp_path.glob("*.partial")) == []


def test_a_target_that_never_verifies_is_named_after_the_wait(tmp_path, monkeypatch):
    target = tmp_path / "case9.m"
    target.write_bytes(b"x")
    sleeps = []

    def replace(src, dst):
        raise PermissionError(5, "held", str(dst))

    def read_bytes(self):
        raise PermissionError(13, "mid-replace", str(self))

    monkeypatch.setattr(cases.os, "replace", replace)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(cases.time, "sleep", sleeps.append)
    with pytest.raises(MatpowerCaseUnavailableError, match="case9.m"):
        cases._write(target, b"data", "0" * 64)
    assert 0 < sum(sleeps) <= cases._REPLACE_WAIT_S + 0.25
    assert list(tmp_path.glob("*.partial")) == []


def test_a_broken_http_response_is_unavailable(tmp_path, monkeypatch):
    def urlopen(url, timeout):
        raise http.client.IncompleteRead(b"func")

    monkeypatch.setattr(cases.urllib.request, "urlopen", urlopen)
    with pytest.raises(MatpowerCaseUnavailableError, match="could not download.*case118"):
        matpower_case("case118", cache_dir=tmp_path)


def test_the_digest_table_is_read_only():
    with pytest.raises(TypeError):
        MATPOWER_CASES["case9"] = "0" * 64


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

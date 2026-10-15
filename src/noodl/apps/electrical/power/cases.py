"""MATPOWER's standard cases, fetched on first use.

The MATPOWER case files are not covered by MATPOWER's BSD licence, so noodl physics does not
redistribute them. `matpower_case(name)` downloads one of the cases below from MATPOWER's
GitHub repository at the commit of release 8.1, checks its SHA-256 and keeps it in a cache
directory; the file is read as text by `read_matpower`, never run. `case9_features` is not a
MATPOWER file: it is `case9` modified for noodl physics (off-nominal taps and phase
shifters, a parallel branch, an out-of-service branch, a bus shunt with conductance, two
generators on one bus, a generator on a PQ bus, an out-of-service generator leaving a PV bus
without generation, and a 10 degree slack angle), derived here from the fetched `case9`.

The cache is `$NOODL_CACHE_DIR/matpower/8.1/`, by default `~/.cache/noodl/matpower/8.1/`.
Without a network connection a case can be placed there by hand: the refusal names the URL,
the path and the expected digest. Digests are of the file's bytes with CRLF line endings
read as LF, so a copy checked out on Windows verifies too.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

MATPOWER_RELEASE = "8.1"
MATPOWER_COMMIT = "1a828c7af590714499284e36ee9c81273388c594"  # tag 8.1
MATPOWER_SOURCE = ("https://raw.githubusercontent.com/MATPOWER/matpower/"
                   f"{MATPOWER_COMMIT}/data/{{name}}.m")

#: SHA-256 of each known case (LF line endings).
MATPOWER_CASES = {
    "case9": "ee50fc7bf9f6019c0f3a3bc94d20978cc667b08f695dc725d00dbd998b358623",
    "case14": "2ffc4e1b734ae6c5e92dbe68b4e36010ed695a4bbcc4d065c74c4fbc39fcf3c1",
    "case30": "3d9030311259b553be85d02336b7e1bcb24ec04775bee6671bdb62d18e4e2137",
    "case118": "bc2e6f22b4b9e776572885ee4b50e4f4ab2ee0c5577e9126e86d906f14c4b5f7",
    "case9_features": "c74d01e06ebc878702db27688a47e4baa61b3fbb085d844f42644a911684fb82",
}
_DERIVED = {"case9_features": "case9"}
_TIMEOUT_S = 30.0


class MatpowerCaseUnavailableError(OSError):
    """A MATPOWER case could not be downloaded (no network, or the server refused)."""


class MatpowerCaseDigestError(ValueError):
    """A downloaded or cached MATPOWER case is not the file whose SHA-256 is recorded."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _cache(cache_dir) -> Path:
    if cache_dir is None:
        root = os.environ.get("NOODL_CACHE_DIR")
        cache_dir = Path(root) if root else Path.home() / ".cache" / "noodl"
    return Path(cache_dir) / "matpower" / MATPOWER_RELEASE


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)


def _row(fields: str) -> str:
    return "\t" + "\t".join(fields.split()) + ";"


_GEN = " 300 -300 {vg} 100 {status} {pmax} 10" + " 0" * 11
_FEATURES_HEADER = """\
function mpc = case9_features
%CASE9_FEATURES  case9 modified to exercise every branch, bus and generator
%   feature the noodl physics MATPOWER reader supports: off-nominal taps, phase
%   shifters, an out-of-service branch, parallel branches, a bus shunt with
%   conductance, two generators on one bus, a generator on a PQ bus, an
%   out-of-service generator leaving a PV bus without generation (MATPOWER then
%   treats the bus as PQ), and a nonzero slack angle.
%   Derived from MATPOWER's case9 (see NOTICE.md in this directory).
"""
#: case9 row -> the rows replacing it in case9_features.
_FEATURES_ROWS = {
    _row("1 3 0 0 0 0 1 1 0 345 1 1.1 0.9"): [_row("1 3 0 0 0 0 1 1 10 345 1 1.1 0.9")],
    _row("5 1 90 30 0 0 1 1 0 345 1 1.1 0.9"): [_row("5 1 90 30 5 10 1 1 0 345 1 1.1 0.9")],
    _row("2 163 6.54" + _GEN.format(vg=1.025, status=1, pmax=300)): [
        _row("2 123 6.54" + _GEN.format(vg=1.025, status=1, pmax=300)),
        _row("2 40 0" + _GEN.format(vg=1.025, status=1, pmax=300))],
    _row("3 85 -10.95" + _GEN.format(vg=1.025, status=1, pmax=270)): [
        _row("3 85 -10.95" + _GEN.format(vg=1.025, status=0, pmax=270)),
        _row("7 60 15" + _GEN.format(vg=1, status=1, pmax=270))],
    _row("1 4 0 0.0576 0 250 250 250 0 0 1 -360 360"): [
        _row("1 4 0 0.0576 0 250 250 250 1.02 -3 1 -360 360")],
    _row("3 6 0 0.0586 0 300 300 300 0 0 1 -360 360"): [
        _row("3 6 0 0.0586 0 300 300 300 0.98 5 1 -360 360")],
    _row("7 8 0.0085 0.072 0.149 250 250 250 0 0 1 -360 360"): [
        _row("7 8 0.0085 0.072 0.149 250 250 250 0 0 1 -360 360"),
        _row("7 8 0.0170 0.144 0.0745 250 250 250 0 0 1 -360 360")],
    _row("9 4 0.01 0.085 0.176 250 250 250 0 0 1 -360 360"): [
        _row("9 4 0.01 0.085 0.176 250 250 250 0 0 0 -360 360")],
}


def _case9_features(case9: str) -> bytes:
    """case9's text with its header replaced, the rows above replaced and its OPF data
    (generator costs) dropped."""
    lines = case9.replace("\r\n", "\n").split("\n")
    body = lines[lines.index("%   MATPOWER") + 1:lines.index("%%-----  OPF Data  -----%%") - 1]
    out = [row for line in body for row in _FEATURES_ROWS.get(line, [line])]
    return (_FEATURES_HEADER + "\n".join(out) + "\n").encode("ascii")


def _refuse_digest(name: str, what: str, expected: str, found: str) -> MatpowerCaseDigestError:
    return MatpowerCaseDigestError(
        f"MATPOWER case {name!r}: {what} has SHA-256 {found}, expected {expected}")


def matpower_case(name: str, *, cache_dir=None) -> Path:
    """The path of MATPOWER 8.1's case `name` (one of `MATPOWER_CASES`), downloaded on first
    use and verified by SHA-256 on every call.

    `cache_dir` overrides the cache root (default `$NOODL_CACHE_DIR`, else
    `~/.cache/noodl`); the file lives in `<root>/matpower/8.1/<name>.m`. Raises
    `MatpowerCaseUnavailableError` (an `OSError`) when the download fails, naming the URL,
    the path to place the file at by hand and its expected digest, and
    `MatpowerCaseDigestError` (a `ValueError`) when a downloaded or cached file has another
    digest; a mismatching download is not cached.
    """
    if name not in MATPOWER_CASES:
        raise ValueError(f"unknown MATPOWER case {name!r}; known cases: "
                         f"{', '.join(MATPOWER_CASES)}")
    expected = MATPOWER_CASES[name]
    path = _cache(cache_dir) / f"{name}.m"
    if path.exists():
        found = _digest(path.read_bytes())
        if found != expected:
            raise _refuse_digest(name, f"the cached file {path} (delete it to download it "
                                 f"again)", expected, found)
        return path
    if name in _DERIVED:
        base = matpower_case(_DERIVED[name], cache_dir=cache_dir)
        data = _case9_features(base.read_text(encoding="ascii"))
        what = f"the file derived from {base}"
    else:
        url = MATPOWER_SOURCE.format(name=name)
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:  # noqa: S310
                data = response.read()
        except OSError as exc:
            raise MatpowerCaseUnavailableError(
                f"could not download MATPOWER case {name!r} from {url} ({exc}); without a "
                f"network connection, save that file as {path} (SHA-256 of its bytes with LF "
                f"line endings: {expected})") from exc
        what = f"the file downloaded from {url}"
    found = _digest(data)
    if found != expected:
        raise _refuse_digest(name, what, expected, found)
    _write(path, data)
    return path


__all__ = [
    "MATPOWER_CASES",
    "MATPOWER_COMMIT",
    "MATPOWER_RELEASE",
    "MatpowerCaseDigestError",
    "MatpowerCaseUnavailableError",
    "matpower_case",
]

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
import http.client
import os
import tempfile
import types
import urllib.request
from pathlib import Path

MATPOWER_RELEASE = "8.1"
MATPOWER_COMMIT = "1a828c7af590714499284e36ee9c81273388c594"  # tag 8.1
MATPOWER_SOURCE = ("https://raw.githubusercontent.com/MATPOWER/matpower/"
                   f"{MATPOWER_COMMIT}/data/{{name}}.m")

#: SHA-256 of each known case (LF line endings), read-only.
MATPOWER_CASES = types.MappingProxyType({
    "case9": "ee50fc7bf9f6019c0f3a3bc94d20978cc667b08f695dc725d00dbd998b358623",
    "case14": "2ffc4e1b734ae6c5e92dbe68b4e36010ed695a4bbcc4d065c74c4fbc39fcf3c1",
    "case30": "3d9030311259b553be85d02336b7e1bcb24ec04775bee6671bdb62d18e4e2137",
    "case118": "bc2e6f22b4b9e776572885ee4b50e4f4ab2ee0c5577e9126e86d906f14c4b5f7",
    "case9_features": "f6abec7d6ef4c675a4214c858fc1db5db1624dcb8d6b9be3cdb0366dfded4857",
})
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


def _write(path: Path, data: bytes, expected: str) -> None:
    """Write `data` to `path` through a uniquely named temporary file, so concurrent
    fetchers of one case never share a partial file. If another fetcher's file is already
    in place and cannot be replaced (Windows), it is kept when its digest verifies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".",
                                       suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        try:
            os.replace(partial, path)
        except (PermissionError, FileExistsError):
            if not (path.exists() and _digest(path.read_bytes()) == expected):
                raise
    finally:
        if os.path.exists(partial):
            os.remove(partial)


# case9_features: case9 with these changes. Columns are 0-based in MATPOWER's tables
# (bus: 4 GS, 5 BS, 8 VA; gen: 0 GEN_BUS, 1 PG, 2 QG, 5 VG, 7 GEN_STATUS; branch: 2 BR_R,
# 3 BR_X, 4 BR_B, 8 TAP, 9 SHIFT, 10 BR_STATUS). Values are the file's text.
_FEATURES_HEADER = """\
function mpc = case9_features
%CASE9_FEATURES  case9 modified to exercise every branch, bus and generator
%   feature the noodl physics MATPOWER reader supports: off-nominal taps, phase
%   shifters, an out-of-service branch, parallel branches, a bus shunt with
%   conductance, two generators on one bus, a generator on a PQ bus, an
%   out-of-service generator leaving a PV bus without generation (MATPOWER then
%   treats the bus as PQ), and a nonzero slack angle.
%   Derived from MATPOWER's case9 (see tests/data/power_flow/NOTICE.md in noodl physics).
"""
#: (table, row) -> {column: new text}
_FEATURES_CHANGES = {
    ("bus", 0): {8: "10"},                          # slack angle 10 degrees
    ("bus", 4): {4: "5", 5: "10"},                  # shunt Gs 5 MW, Bs 10 MVAr at bus 5
    ("gen", 1): {1: "123"},                         # bus 2's generation split in two ...
    ("gen", 2): {7: "0"},                           # bus 3's only generator out of service
    ("branch", 0): {8: "1.02", 9: "-3"},            # tap and phase shift on 1-4
    ("branch", 3): {8: "0.98", 9: "5"},             # tap and phase shift on 3-6
    ("branch", 8): {10: "0"},                       # 9-4 out of service
}
#: (table, row) -> rows inserted after it, each a copy of that row with {column: new text}
_FEATURES_INSERTED = {
    ("gen", 1): [{1: "40", 2: "0"}],                # ... the second generator on bus 2
    ("gen", 2): [{0: "7", 1: "60", 2: "15", 5: "1"}],   # a generator on the PQ bus 7
    ("branch", 5): [{2: "0.0170", 3: "0.144", 4: "0.0745"}],  # a second 7-8 line
}


def _edited(fields: list[str], change: dict[int, str]) -> str:
    out = list(fields)
    for column, text in change.items():
        out[column] = text
    return "\t" + "\t".join(out) + ";"


def _case9_features(case9: str) -> bytes:
    """case9's text with its header replaced, the changes above applied and its OPF data
    (generator costs) dropped."""
    lines = case9.replace("\r\n", "\n").split("\n")
    body = lines[lines.index("%   MATPOWER") + 1:lines.index("%%-----  OPF Data  -----%%") - 1]
    out: list[str] = []
    table, row = None, 0
    for line in body:
        if line.startswith("mpc.") and line.endswith("= ["):
            table, row = line[4:line.index(" ")], 0
            out.append(line)
        elif line == "];":
            table = None
            out.append(line)
        elif table is not None:
            fields = line.strip().rstrip(";").split("\t")
            out.append(_edited(fields, _FEATURES_CHANGES.get((table, row), {})))
            out.extend(_edited(fields, c) for c in _FEATURES_INSERTED.get((table, row), []))
            row += 1
        else:
            out.append(line)
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
        except (OSError, http.client.HTTPException) as exc:
            raise MatpowerCaseUnavailableError(
                f"could not download MATPOWER case {name!r} from {url} ({exc}); without a "
                f"network connection, save that file as {path} (SHA-256 of its bytes with LF "
                f"line endings: {expected})") from exc
        what = f"the file downloaded from {url}"
    found = _digest(data)
    if found != expected:
        raise _refuse_digest(name, what, expected, found)
    _write(path, data, expected)
    return path


__all__ = [
    "MATPOWER_CASES",
    "MATPOWER_COMMIT",
    "MATPOWER_RELEASE",
    "MatpowerCaseDigestError",
    "MatpowerCaseUnavailableError",
    "matpower_case",
]

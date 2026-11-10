"""Golden-file helpers: JSON storage of reference numeric results for regression tests."""

import json
from pathlib import Path
from typing import Any

import torch

_GOLDEN_DIR = Path(__file__).parent


def _path(name: str) -> Path:
    return _GOLDEN_DIR / f"{name}.json"


def load_golden(name: str) -> dict[str, Any]:
    """Load the golden data stored under `name`; raises FileNotFoundError if absent."""
    with _path(name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_golden(name: str, data: dict[str, Any]) -> None:
    """Overwrite the golden data stored under `name` with `data`."""
    with _path(name).open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def ser(x: Any) -> Any:
    """A JSON-ready, exactly round-tripping form of tensors, numbers, dicts and lists."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.is_complex():
            data = torch.view_as_real(t).tolist()
        else:
            data = t.tolist()
        return {"dtype": str(t.dtype), "shape": list(t.shape), "data": data}
    if isinstance(x, dict):
        return {str(k): ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ser(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    raise TypeError(f"cannot record {type(x).__name__}: {x!r}")


def grads_of(loss: torch.Tensor, params: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Back-propagate `loss`; the gradients of `params` (None where nothing reached)."""
    for p in params.values():
        p.grad = None
    loss.backward()
    return {k: p.grad for k, p in params.items()}

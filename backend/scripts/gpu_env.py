"""Run SOLUM code on the global interpreter, which has CUDA torch.

The problem this works around: two unrelated projects are installed on
the global interpreter as *editable* packages and both claim the
top-level name `app`. Editable installs register a `meta_path` finder,
and `meta_path` is consulted before `sys.path`, so no amount of
`sys.path.insert` makes this project win - Python imports the other
project's `app` and this one appears to have no code at all.

`-S` skips site initialisation, so the `.pth` files that install those
finders are never processed. Adding site-packages back afterwards gets us
torch (with CUDA) and transformers without re-registering the finders.

No packages are installed and nothing is downloaded. That is the point:
the CUDA build of torch already exists on this machine and the venv was
paying for a second, CPU-only copy of it.

Usage:
    D:\\Python\\python.exe -S scripts\\gpu_env.py
"""

from __future__ import annotations

import site
import sys
import sysconfig
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
EVAL = BACKEND.parent / "eval"


def _purelib() -> str:
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path and Path(path).is_dir():
            return path
    return ""


def setup() -> None:
    """Put site-packages on the path and this project ahead of it."""
    # This project first, so `app` resolves here.
    for entry in (str(BACKEND), str(EVAL)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    # Then the global packages, so torch and transformers are importable.
    # Adding the directory does not re-run its .pth files, which is the
    # whole point: the editable finders stay uninstalled.
    for purelib in filter(None, [_purelib(), site.getsitepackages()[-1:] and site.getsitepackages()[-1]]):
        if purelib not in sys.path:
            sys.path.append(purelib)


def assert_ours() -> None:
    """Fail loudly if another project's `app` won the import."""
    import app

    origin = getattr(app, "__path__", [None])[0] if hasattr(app, "__path__") else None
    if origin is None:
        origin = getattr(app, "__file__", None)
    if origin and BACKEND not in Path(str(origin)).resolve().parents:
        raise SystemExit(
            "Another project's `app` package won the import:\n"
            f"  got      {origin}\n"
            f"  expected under {BACKEND}\n"
            "This means site-packages .pth files were processed. Run with -S."
        )


if __name__ == "__main__":
    setup()
    assert_ours()
    import app.services.embeddings as emb

    print("app resolves to :", list(emb.__file__ and [emb.__file__])[0])
    import torch

    print("torch           :", torch.__version__)
    print("cuda available  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device          :", torch.cuda.get_device_name(0))

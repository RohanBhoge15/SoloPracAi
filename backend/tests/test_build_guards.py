"""Guard against two failure modes that have actually bitten this project.

1. A file that parses on Windows but fails inside the Linux container.
   Caused by CRLF endings interacting with line-continuation backslashes,
   or by a truncated write. The Dockerfile has a build-time import check,
   but that only runs on `docker build`; this runs in the test suite.

2. A module that silently fails to import, so a route 404s rather than
   erroring loudly. Worth asserting on the things the app cannot work
   without.
"""

from __future__ import annotations

import ast
import io
import pathlib
from pathlib import Path
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("app", "scripts", "tests")


def _python_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for name in SOURCE_DIRS:
        directory = ROOT / name
        if directory.exists():
            out.extend(directory.rglob("*.py"))
    return out


def test_every_source_file_parses_from_bytes():
    """Parse raw bytes, exactly as the container will.

    Reading with newline='' on Windows silently normalises CRLF, which
    hides the class of bug where a file is valid locally and broken in
    the image. Parsing bytes does not normalise anything.
    """
    broken: list[str] = []
    for path in _python_files():
        try:
            ast.parse(path.read_bytes())
        except SyntaxError as exc:
            broken.append(f"{path.relative_to(ROOT)}:{exc.lineno} {exc.msg}")
    assert not broken, "files that parse differently outside Windows:\n" + "\n".join(
        broken
    )


def test_no_continuation_backslash_before_crlf():
    """A trailing backslash before CRLF escapes the carriage return.

    The lines then concatenate and a triple-quoted string can silently
    swallow the rest of the module. This exact bug cost a full rebuild
    cycle once already.
    """
    offenders: list[str] = []
    for path in _python_files():
        raw = path.read_bytes()
        if b"\\\r\n" in raw:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "trailing backslash before CRLF in: " + ", ".join(offenders)
    )


def test_every_module_compiles():
    """Byte-compile every source file, not only those a test imports.

    A module no test imports can carry a syntax error straight into the
    image and only surface when its route is first called. This caught a
    stray backslash inside a prompt string that had passed review and
    failed only inside the container.
    """
    broken: list[str] = []
    for path in _python_files():
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as exc:
            broken.append(f"{path.relative_to(ROOT)}:{exc.lineno} {exc.msg}")
        except ValueError as exc:
            broken.append(f"{path.relative_to(ROOT)} {exc}")
    assert not broken, "files that do not compile:\n" + "\n".join(broken)


def test_optional_model_imports_stay_lazy():
    """An optional model must degrade to a missing feature, not a crash.

    llama-cpp-python comes from a prebuilt wheel and is allowed to be
    absent. Importing it at module scope would take the whole API down
    over a feature that is off by default.
    """
    source = (ROOT / "app/services/vision_llm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("llama_cpp"), (
                    "llama_cpp must be imported inside the loader, not at "
                    "module scope"
                )
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("llama_cpp"), (
                "llama_cpp must be imported inside the loader, not at "
                "module scope"
            )


def test_no_duplicate_table_definitions():
    """Two models claiming one table name break create_all at import."""
    seen: dict[str, list[str]] = {}
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and any(
                            isinstance(t, ast.Name) and t.id == "__tablename__"
                            for t in stmt.targets
                        )
                        and isinstance(stmt.value, ast.Constant)
                    ):
                        seen.setdefault(stmt.value.value, []).append(
                            f"{path.relative_to(ROOT)}:{node.name}"
                        )

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    assert not duplicates, f"tables defined more than once: {duplicates}"


def test_core_modules_import():
    """The modules the app cannot run without must import cleanly."""
    import importlib

    for name in (
        "app.core.validity",
        "app.core.config",
        "app.core.security",
        "app.services.retrieval",
        "app.services.provenance",
        "app.services.pipeline",
        "app.services.pii",
        "app.services.pdf",
        "app.services.storage",
        "app.services.notifications",
        "app.identity.linker",
        "app.documents.render",
        "app.documents.templates",
        "app.documents.weekly",
    ):
        importlib.import_module(name)


def test_research_core_is_importable_and_pure():
    """The contribution module must not pull in infrastructure.

    If validity.py ever imports a database driver, the TAP-QA harness can
    no longer run the baselines without a live Postgres, and the ability
    to evaluate the contribution independently is lost.
    """
    source = (ROOT / "app/core/validity.py").read_text(encoding="utf-8")
    for forbidden in ("sqlalchemy", "fastapi", "httpx", "redis", "qdrant"):
        assert forbidden not in source, (
            f"validity.py must stay infrastructure-free; found {forbidden}"
        )


def test_compose_services_reference_modules_that_exist():
    """A service whose command names a missing module crash-loops forever.

    `solum-worker` ran `python -m app.worker` for the whole project with no
    such file, and the container sat in a restart loop that nobody looked
    at, because the API worked and nothing else reported a problem.

    This parses docker-compose.yml and checks that every
    `python -m <module>` command resolves to a real file. It is a text
    check rather than a YAML parse on purpose: PyYAML is not a dependency,
    and the pattern is narrow enough to read.
    """
    compose = ROOT.parent / "docker-compose.yml"
    if not compose.is_file():
        pytest.skip("docker-compose.yml not present")

    text = compose.read_text(encoding="utf-8")
    missing: list[str] = []

    for match in re.finditer(r"python\s+-m\s+([\w.]+)", text):
        module = match.group(1)
        relative = Path(*module.split(".")).with_suffix(".py")
        if not (ROOT / relative).is_file():
            missing.append(f"{module} (expected {relative})")

    assert not missing, (
        "docker-compose runs modules that do not exist: "
        + ", ".join(missing)
    )


def test_compose_mounts_the_model_directory_read_only():
    """The app derives encoder configs and must not write into /models.

    Writing there failed at load time with "Read-only file system" and the
    encoder silently never came up, so the mount mode is load-bearing
    rather than tidiness.
    """
    compose = ROOT.parent / "docker-compose.yml"
    if not compose.is_file():
        pytest.skip("docker-compose.yml not present")
    text = compose.read_text(encoding="utf-8")
    assert ":/models:ro" in text, (
        "the model directory must be mounted read-only; derived configs "
        "belong in EMBEDDER_CACHE"
    )
    assert "EMBEDDER_CACHE" in text, (
        "a writable cache path must be configured for derived encoder configs"
    )

"""Every module and script of the three repos imports, and every built-in `--check` self-test
passes. Importing runs a file's top level only: scripts keep their work under
`if __name__ == "__main__"`, so nothing here loads the model, touches the GPU or writes a file.
The Space app (space/app.py) is left out: gradio is the Space's own dependency, not this repo's."""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = {  # import root -> directories whose files are imported as package modules
    ROOT: ("audit", "interventions"),
    ROOT.parent / "tiny-aya-circuits": ("aya_circuits",),
    ROOT.parent / "nothing-else-broke": ("neb",),
}
SCRIPT_DIRS = (ROOT / "checks", ROOT / "data")  # plain script folders, imported by path
SELF_CHECKS = (
    ["-m", "aya_circuits.certify", "--check"],
    ["interventions/run_a20.py", "--check"],
    ["interventions/run_a21.py", "--check"],
    ["audit/a21_analyze.py", "--check"],
    ["data/a22_build.py", "--check"],
    ["interventions/run_a22.py", "--check"],
    ["audit/a22_analyze.py", "--check"],
    ["data/a23_build.py", "--check"],
    ["audit/a23_fluency.py", "--check"],
)


def module_names() -> list[str]:
    return [
        ".".join(p.relative_to(root).with_suffix("").parts)
        for root, dirs in PACKAGES.items()
        for d in dirs
        for p in sorted((root / d).glob("*.py"))
        if p.name != "__init__.py"
    ]


def script_paths() -> list[Path]:
    return [p for d in SCRIPT_DIRS for p in sorted(d.glob("*.py"))]


@pytest.mark.parametrize("name", module_names())
def test_module_imports(name: str) -> None:
    sys.path.insert(0, str(ROOT))
    importlib.import_module(name)


@pytest.mark.parametrize("path", script_paths(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_imports(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"_script_{path.parent.name}_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize("args", SELF_CHECKS, ids=lambda a: " ".join(a))
def test_self_check(args: list[str]) -> None:
    r = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.strip().endswith("ok"), r.stdout[-500:]

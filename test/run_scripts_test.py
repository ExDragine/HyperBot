# ruff: noqa: S101

from pathlib import Path


def test_run_sh_matches_run_ps1_startup_contract():
    repo_root = Path(__file__).resolve().parents[1]
    run_sh = (repo_root / "run.sh").read_text(encoding="utf-8")
    run_ps1 = (repo_root / "run.ps1").read_text(encoding="utf-8")

    assert "source .venv/bin/activate" in run_sh
    assert ".venv/Scripts/Activate.ps1" in run_ps1
    assert "uv run --locked nb run" in run_sh
    assert "uv run --locked nb run" in run_ps1
    assert "sleep 5" in run_sh
    assert "sleep 5" in run_ps1

    assert "uv sync" not in run_sh
    assert "playwright install" not in run_sh
    assert "nb run" not in run_sh.replace("uv run --locked nb run", "")

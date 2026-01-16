import json
import subprocess


def _dry_run(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_fig2_paper_defaults_dry_run():
    data = _dry_run(
        [
            "python",
            "examples/prl2018_figures.py",
            "fig2_single",
            "--paper-defaults",
            "--dry-run",
        ]
    )

    assert data["ks"]["L"] == 22.0
    assert data["ks"]["Q"] == 64
    assert data["ks"]["mu"] == 0.0
    assert data["ks"]["dt"] == 0.25
    assert data["rc"]["g"] == 1
    assert data["rc"]["l"] == 0
    assert data["rc"]["q"] == 64


def test_fig4_paper_defaults_dry_run():
    data = _dry_run(
        [
            "python",
            "examples/prl2018_figures.py",
            "fig4_parallel",
            "--paper-defaults",
            "--dry-run",
        ]
    )

    assert data["ks"]["L"] == 200.0
    assert data["ks"]["Q"] == 512
    assert data["ks"]["mu"] == 0.01
    assert data["ks"]["lam"] == 100.0
    assert data["ks"]["dt"] == 0.25
    assert data["rc"]["g"] == 64
    assert data["rc"]["l"] == 6
    assert data["rc"]["q"] == 8
    min_eval = data["eval"]["K"] * data["eval"]["tau"] + data["eval"]["epsilon"] + 1
    assert data["eval"]["T_eval"] >= min_eval


def test_fig6_paper_defaults_dry_run():
    data = _dry_run(
        [
            "python",
            "examples/prl2018_figures.py",
            "fig6_shared_weights",
            "--paper-defaults",
            "--dry-run",
        ]
    )

    assert sorted(data["fig6"]["mu_values"]) == [0.0, 0.01]


def test_fig2_paper_defaults_quick_preserves_single():
    data = _dry_run(
        [
            "python",
            "examples/prl2018_figures.py",
            "fig2_single",
            "--paper-defaults",
            "--quick",
            "--dry-run",
        ]
    )

    assert data["rc"]["g"] == 1
    assert data["rc"]["l"] == 0

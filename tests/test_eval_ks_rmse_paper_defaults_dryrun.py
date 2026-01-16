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


def test_eval_paper_defaults_dry_run():
    data = _dry_run(
        [
            "python",
            "examples/eval_ks_rmse.py",
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
    assert data["rc"]["hidden_size"] == 5000
    assert data["rc"]["spectral_radius"] == 0.6
    assert data["rc"]["sigma"] == 1.0
    assert data["rc"]["kappa"] == 3
    assert data["eval"]["K"] == 30
    assert data["eval"]["tau"] == 1000
    assert data["eval"]["epsilon"] == 10
    assert data["eval"]["n_trials"] == 10
    assert data["derived"]["required_eval"] == (
        data["eval"]["K"] * data["eval"]["tau"] + data["eval"]["epsilon"] + 1
    )
    assert data["eval"]["T_eval"] >= data["derived"]["required_eval"]
    assert data["rc"]["q"] == 8
    assert abs(data["rc"]["effective_density"] - 3 / 5000) < 1e-9


def test_eval_paper_defaults_quick_dry_run():
    data = _dry_run(
        [
            "python",
            "examples/eval_ks_rmse.py",
            "--paper-defaults",
            "--quick",
            "--dry-run",
        ]
    )

    assert data["ks"]["L"] == 200.0
    assert data["ks"]["mu"] == 0.01
    assert data["ks"]["lam"] == 100.0
    assert data["ks"]["dt"] == 0.25
    assert data["rc"]["q"] == data["ks"]["Q"] // data["rc"]["g"]
    assert data["eval"]["T_eval"] >= data["derived"]["required_eval"]

import json
import re
import subprocess
from pathlib import Path


def test_eval_ks_rmse_quick(tmp_path):
    out_dir = tmp_path / "ks_eval_test"
    result = subprocess.run(
        [
            "python",
            "examples/eval_ks_rmse.py",
            "--quick",
            "--out-dir",
            str(out_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    assert re.search(r"RMSE.*\d", result.stdout)

    rmse_png = out_dir / "rmse_curve.png"
    rmse_json = out_dir / "rmse_curve.json"
    assert rmse_png.exists()
    assert rmse_json.exists()

    data = json.loads(rmse_json.read_text(encoding="utf-8"))
    rmse_curve = data["rmse_curve"]
    assert rmse_curve
    assert all(value == value for value in rmse_curve)

import re
import subprocess
from pathlib import Path


def _run_fig(command, out_dir):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    output = result.stdout
    assert re.search(r"RMSE.*\d", output)
    pngs = list(Path(out_dir).glob("*.png"))
    assert pngs


def test_prl2018_figures_quick(tmp_path):
    out_dir = tmp_path / "prl2018_figs"
    _run_fig(
        [
            "python",
            "examples/prl2018_figures.py",
            "fig4_parallel",
            "--quick",
            "--out-dir",
            str(out_dir),
        ],
        out_dir,
    )
    _run_fig(
        [
            "python",
            "examples/prl2018_figures.py",
            "fig6_shared_weights",
            "--quick",
            "--out-dir",
            str(out_dir),
        ],
        out_dir,
    )

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("examples/prl2018_figures.py")
OUTPUT_DIR = Path("examples/figures/prl2018")


@pytest.mark.parametrize(
    "command",
    ["fig2_single", "fig4_parallel", "fig5_scaling", "fig6_shared_weights"],
)
def test_prl2018_figures_quick(command):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), command, "--quick"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "RMSE" in result.stdout
    output_file = OUTPUT_DIR / f"{command}.png"
    assert output_file.exists()

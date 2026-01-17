import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_run_script_reports_short_test_data(tmp_path: Path):
    data_path = tmp_path / "ks_short.npz"
    train_u = np.zeros((100, 4), dtype=np.float32)
    test_u = np.zeros((20, 4), dtype=np.float32)
    meta = {"train_steps": 100}
    meta_json = json.dumps(meta, sort_keys=True)
    np.savez(data_path, train_u=train_u, test_u=test_u, meta_json=meta_json)

    cmd = [
        sys.executable,
        "examples/ks/run_parallel_rc_ks.py",
        "--data",
        str(data_path),
        "--predict-length",
        "10",
        "--sync-length",
        "5",
        "--num-intervals",
        "3",
        "--interval-stride",
        "10",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "need at least 35 steps, got 20" in result.stderr

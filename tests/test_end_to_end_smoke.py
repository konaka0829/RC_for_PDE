import subprocess
import sys
from pathlib import Path


def test_end_to_end_smoke(tmp_path: Path):
    data_path = tmp_path / "ks_demo.npz"

    gen_cmd = [
        sys.executable,
        "examples/ks/generate_ks_dataset.py",
        "--out",
        str(data_path),
        "--L",
        "30",
        "--Q",
        "16",
        "--dt",
        "0.25",
        "--mu",
        "0.01",
        "--lambda",
        "15",
        "--total-steps",
        "800",
        "--train-steps",
        "600",
        "--seed",
        "0",
    ]
    subprocess.run(gen_cmd, check=True)

    run_cmd = [
        sys.executable,
        "examples/ks/run_parallel_rc_ks.py",
        "--data",
        str(data_path),
        "--g",
        "4",
        "--l",
        "1",
        "--reservoir-size-approx",
        "120",
        "--degree",
        "3",
        "--spectral-radius",
        "0.6",
        "--sigma",
        "1.0",
        "--beta",
        "1e-4",
        "--train-length",
        "600",
        "--predict-length",
        "50",
        "--sync-length",
        "5",
        "--num-intervals",
        "2",
        "--interval-stride",
        "50",
        "--chunk-size",
        "64",
        "--seed",
        "0",
    ]
    subprocess.run(run_cmd, check=True)

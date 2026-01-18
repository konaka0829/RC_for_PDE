import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestKSCLI(unittest.TestCase):
    def test_cli_quick(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "ks_quick.npz"
            result = subprocess.run(
                [
                    sys.executable,
                    "examples/ks_generate.py",
                    "--quick",
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = np.load(output_path, allow_pickle=True)
            u = data["u"]
            self.assertEqual(u.shape, (300, 64))
            self.assertTrue(np.isfinite(u).all())
            self.assertEqual(data["x"].shape, (64,))
            self.assertEqual(data["t"].shape, (300,))
            for suffix in (".png", ".pdf", ".svg"):
                self.assertTrue(output_path.with_suffix(suffix).exists())


if __name__ == "__main__":
    unittest.main()

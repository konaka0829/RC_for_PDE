import unittest

import numpy as np

from torchesn.pde import simulate_ks_etdrk4


class TestKSETDRK4(unittest.TestCase):
    def test_shapes_and_finite(self):
        result = simulate_ks_etdrk4(
            n_steps=2000,
            dt=0.25,
            n_grid=64,
            domain_length=22,
            transient=100,
            save_every=1,
            seed=0,
        )
        u = result["u"]
        self.assertEqual(u.shape, (2000, 64))
        self.assertTrue(np.isfinite(u).all())
        self.assertLess(np.max(np.abs(u)), 1e3)

    def test_deterministic_seed(self):
        run1 = simulate_ks_etdrk4(n_steps=50, seed=123)
        run2 = simulate_ks_etdrk4(n_steps=50, seed=123)
        self.assertTrue(np.array_equal(run1["u"], run2["u"]))

    def test_odd_grid_raises(self):
        with self.assertRaises(ValueError):
            simulate_ks_etdrk4(n_steps=10, n_grid=63)


if __name__ == "__main__":
    unittest.main()

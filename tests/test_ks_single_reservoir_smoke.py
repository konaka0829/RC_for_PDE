import unittest

from examples.ks_single_reservoir import KSRunConfig, run_experiment


class KSSingleReservoirSmokeTests(unittest.TestCase):
    def test_single_reservoir_smoke(self):
        config = KSRunConfig(
            n_grid=16,
            train_steps=300,
            pred_steps=50,
            warmup_steps=50,
            hidden_size=200,
            lambda_reg=1e-3,
            seed=0,
            save_plots=False,
        )
        rmse = run_experiment(config)
        self.assertTrue(rmse == rmse)
        self.assertTrue(rmse < float("inf"))


if __name__ == "__main__":
    unittest.main()

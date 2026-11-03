"""Check reproducible sampling and reuse of the GPU trial launcher."""
import random
import unittest
from unittest.mock import patch

from src.training.random_search import run_random_search, sample_parameters


class RandomSearchTests(unittest.TestCase):
    def test_seeded_samples_stay_in_ranges_without_changing_global_rng(self):
        state = random.getstate()
        samples = sample_parameters(4, [0.0005, 0.001], [0.05, 0.1], 42)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(samples, sample_parameters(4, [0.0005, 0.001], [0.05, 0.1], 42))
        self.assertNotEqual(samples, sample_parameters(4, [0.0005, 0.001], [0.05, 0.1], 43))
        self.assertEqual(len(samples), 4)
        for rate, dropout in samples:
            self.assertTrue(0.0005 <= rate <= 0.001)
            self.assertTrue(0.05 <= dropout <= 0.1)

    def test_exact_trial_budget_and_metadata_reach_shared_launcher(self):
        config = {'model': 'bird_cnn_delayed_pooling', 'seed': 42}
        with patch('src.training.random_search.run_trials') as launch:
            run_random_search(config, [0, 1], 120, 4, [0.0005, 0.001], [0.05, 0.1],
                              search_seed=7, dry_run=True)
        args, kwargs = launch.call_args
        self.assertEqual(args[:3], (config, [0, 1], 120))
        self.assertEqual(len(args[3]), 4)
        self.assertEqual(kwargs, dict(method='random', dry_run=True, search_seed=7))
        self.assertEqual(config['seed'], 42)

    def test_bad_counts_and_ranges_are_rejected(self):
        for count, rates, dropouts in [(0, [0.001, 0.01], [0, 0.1]),
                                      (4, [0.01, 0.001], [0, 0.1]),
                                      (4, [0, 0.001], [0, 0.1]),
                                      (4, [0.001, float('nan')], [0, 0.1]),
                                      (4, [0.001, 0.01], [0.1, 0.05]),
                                      (4, [0.001, 0.01], [0, 1])]:
            with self.assertRaises(ValueError):
                sample_parameters(count, rates, dropouts, 42)


if __name__ == '__main__':
    unittest.main()

"""Sample independent learning rates and convolutional-dropout probabilities."""
import math
import random

from src.training.grid_search import run_trials


def sample_parameters(trials, lr_range, dropout2d_range, seed):
    if trials < 1:
        raise ValueError("trials must be positive.")
    if len(lr_range) != 2 or not all(math.isfinite(x) for x in lr_range):
        raise ValueError("lr-range needs two finite bounds.")
    if not 0 < lr_range[0] <= lr_range[1]:
        raise ValueError("lr-range must satisfy 0 < MIN <= MAX.")
    if len(dropout2d_range) != 2 or not all(math.isfinite(x) for x in dropout2d_range):
        raise ValueError("dropout2d-range needs two finite bounds.")
    if not 0 <= dropout2d_range[0] <= dropout2d_range[1] < 1:
        raise ValueError("dropout2d-range must satisfy 0 <= MIN <= MAX < 1.")

    rng = random.Random(seed)  # Does not change the model/data split seed.
    return [
        (math.exp(rng.uniform(math.log(lr_range[0]), math.log(lr_range[1]))),
         rng.uniform(*dropout2d_range))
        for _ in range(trials)
    ]


def run_random_search(config, gpus, epochs, trials, lr_range, dropout2d_range,
                      search_seed=42, dry_run=False):
    combinations = sample_parameters(trials, lr_range, dropout2d_range, search_seed)
    run_trials(config, gpus, epochs, combinations, method="random",
               dry_run=dry_run, search_seed=search_seed)

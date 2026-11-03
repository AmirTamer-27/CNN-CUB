"""Launch separate training processes, with at most one trial per GPU."""
from collections import deque
from datetime import datetime
from itertools import product
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import torch
import yaml


def run_grid_search(config, gpus, epochs, learning_rates, dropout2d_values,
                    dry_run=False):
    run_trials(config, gpus, epochs, list(product(learning_rates, dropout2d_values)),
               method="grid", dry_run=dry_run)


def run_trials(config, gpus, epochs, combinations, method, dry_run=False,
               search_seed=None):
    """Common GPU queue used by grid and random search."""
    learning_rates = [rate for rate, _ in combinations]
    dropout2d_values = [dropout for _, dropout in combinations]
    if config.get("model") != "bird_cnn_delayed_pooling":
        raise ValueError("This search supports Dropout2d on bird_cnn_delayed_pooling only.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("Provide distinct, nonnegative GPU indices.")
    if not learning_rates or any(not math.isfinite(x) or x <= 0 for x in learning_rates):
        raise ValueError("Learning rates must be finite and positive.")
    if not dropout2d_values or any(not math.isfinite(x) or not 0 <= x < 1 for x in dropout2d_values):
        raise ValueError("Dropout2d values must be in [0, 1).")

    search_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    trials = deque()
    for index, (rate, dropout) in enumerate(combinations, 1):
        trial = dict(config)
        trial.update(
            learning_rate=rate, dropout2d=dropout, epochs=epochs,
            run_name=f"{method}_{search_id}_{index}_lr{rate:g}_d2d{dropout:g}_{epochs}ep",
            search_id=search_id, search_method=method, search_trial=index,
        )
        if method == "grid":
            trial.update(grid_search_id=search_id, grid_trial=index)
        if search_seed is not None:
            trial["search_seed"] = search_seed
        trials.append(trial)
        print(f"Trial {index}: LR={rate:g}, Dropout2d={dropout:g}, epochs={epochs}", flush=True)
    print(f"At most {len(gpus)} trials at once; one per GPU: {gpus}", flush=True)
    if dry_run:
        return

    if not torch.cuda.is_available() or max(gpus) >= torch.cuda.device_count():
        raise ValueError("Requested GPUs unavailable. Select T4 x2 or use --gpus 0 for one GPU.")
    # Respect an existing CUDA mask: arguments refer to GPUs visible to this parent.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    output_dir = Path(f"outputs/{method}_search") / search_id
    output_dir.mkdir(parents=True)
    running = {}
    failures = []
    try:
        while trials or running:
            for gpu in gpus:
                if not trials or gpu in running:
                    continue
                trial = trials.popleft()
                config_path = output_dir / f"{trial['run_name']}.yaml"
                log_path = output_dir / f"{trial['run_name']}.log"
                with config_path.open("w") as file:
                    yaml.safe_dump(trial, file, sort_keys=False)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = devices[gpu].strip()
                env["OMP_NUM_THREADS"] = "2"
                env["MKL_NUM_THREADS"] = "2"
                env.pop("MLFLOW_RUN_ID", None)
                log_file = log_path.open("w")
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-u", "-m", "src.cli", "train", "--config", str(config_path)],
                        env=env, stdout=log_file, stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log_file.close()
                    raise
                running[gpu] = (process, log_file, trial["run_name"])
                print(f"Started on GPU {gpu}: {trial['run_name']}\n  Live log: {log_path}", flush=True)

            for gpu, (process, log_file, name) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log_file.close()
                del running[gpu]
                print(f"{'Finished' if code == 0 else 'FAILED'} on GPU {gpu}: {name} (exit {code})", flush=True)
                if code != 0:
                    failures.append(name)
            if running:
                time.sleep(1)
    finally:
        # Interrupting the search must also stop its training processes.
        for process, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
        for process, log_file, _ in running.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_file.close()

    if failures:
        raise RuntimeError(f"{len(failures)} trial(s) failed; inspect logs in {output_dir}: {failures}")
    print(f"All trials finished. Compare search_id={search_id} in MLflow tags.", flush=True)

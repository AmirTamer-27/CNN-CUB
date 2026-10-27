# Project structure and training flow

## What goes where

| Location | Purpose |
| --- | --- |
| `configs/classification_scratch.yaml` | Experiment settings: image size, batch size, epochs, learning rate, and seed. |
| `src/cli.py` | Starting point: reads settings and connects the other files. |
| `src/data/dataset.py` | Loads the Hugging Face training data, creates training/validation subsets, and builds DataLoaders. |
| `src/data/transforms.py` | Resizes and pads images, then converts them into tensors. |
| `src/models/bird_cnn.py` | Defines the CNN layers and forward pass. |
| `src/training/trainer.py` | Runs training batches, validation, and the epoch loop. |
| `src/evaluation/metrics.py` | Calculates loss, accuracy, and macro recall. |
| `notebooks/` | Image inspection, EDA, and interactive experiments. |
| `requirements.txt` | Python dependencies. |

## Start training

From the repository root, with dependencies installed:

```bash
python -m src.cli train --config configs/classification_scratch.yaml
```

The same command can run on Kaggle after cloning the repository.

## How execution moves through the files

1. **`cli.py`:** reads the YAML configuration, sets the PyTorch seed, and chooses CUDA, MPS, or CPU.
2. **`dataset.py`:** `make_loaders(config)` creates a seeded, stratified training/validation split and returns the DataLoaders and class names.
3. **`cli.py`:** creates a fresh `BirdCNN`, cross-entropy loss, and Adam optimizer, then calls `fit(...)`.
4. **`trainer.py`:** `fit` repeats the following for each epoch:
   - `train_one_epoch`: requests batches, predicts classes, calculates loss, and updates weights.
   - `validate`: measures performance without updating weights or BatchNorm statistics.
   - Records metrics in memory and prints progress.
5. **During data loading:** `with_transform` calls the preparation function in `transforms.py` when samples are requested. The DataLoader stacks their tensors into batches.
6. **After training:** `fit` returns the metric history. The CLI currently does not capture that return value.

The official test dataset is not loaded during training. No result files or model checkpoints are saved, and MLflow logging is not connected yet.

## Trying another experiment

- **Different settings:** copy the YAML file, change the setting you want to investigate, and run the command with that file's path. Each invocation starts fresh weights.
- **Different architecture:** edit the CNN or add another model file, then update model selection in `cli.py`. Changing the model name in YAML alone does not implement it.
- **Comparing results:** keep the data split fixed and compare validation metrics. Commit code changes so each experiment can be tied to its code version.

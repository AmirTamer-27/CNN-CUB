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
6. **After training:** `fit` returns the metric history, and the CLI uploads each epoch's metrics to the MLflow run.

The official test dataset is not loaded during training. The CLI saves the final model state in `<run_name>.pth` and uploads it to the MLflow run under `weights/`, alongside settings and metric history. Credentials come from the environment.

## Analyze a saved model

```bash
python -m src.cli analyze-validation --config configs/classification_scratch_delayed_pooling.yaml --weights model_weights.pth
```

The config and architecture code must match the weights. The command recreates
the validation split and loads the trained state. The analysis file uses the
per-image loss/prediction loop followed by Grad-CAM for selected images. Two
three-column grids show the original, true-class overlay and predicted-class overlay:
`highest_loss_gradcam.png` (10 highest losses) and `random_predictions_gradcam.png`
(5 random correct + 5 random incorrect, excluding the top-loss group).
If a group is too small, remaining samples fill the grid when available. Files are
saved under `outputs/validation_analysis/`. Display them in a notebook after the CLI.
No weight updates, overall accuracy, first-five preview or MLflow connection are involved.

## Trying another experiment

- **Different settings:** copy the YAML file, change the setting you want to investigate, and run the command with that file's path. Each invocation starts fresh weights.
- **Different architecture:** edit the CNN or add another model file, then update model selection in `cli.py`. Changing the model name in YAML alone does not implement it.
- **Comparing results:** keep the data split fixed and compare validation metrics. Commit code changes so each experiment can be tied to its code version.

## Features from multiple depths

Run `python -m src.cli train --config configs/classification_scratch_multidepth.yaml`
to compare against the 320px delayed-pooling experiment with the same settings.
`src/models/bird_cnn_multidepth.py` reuses its six blocks, globally averages
outputs from blocks 4, 5, and 6, and concatenates their 128, 256, and 512 features.
Dropout and a linear layer map these 896 values to species scores. For saved-weight
analysis, use this same config and the multi-depth run's weights.

## Residual experiment with three stages

Run `python -m src.cli train --config configs/classification_scratch_residual.yaml`.
`src/models/bird_cnn_residual.py` defines one residual block and a six-block CNN.
Each block adds its convolution path to a shortcut, then applies ReLU.
A 3x3 stem convolution produces 32 channels without downsampling. Blocks 1–2
use 32 channels, blocks 3–4 use 64, and blocks 5–6 use 128. Blocks 3 and 5
use stride 2 in both the first main-path convolution and the 1x1 shortcut.
The other four blocks use identity shortcuts. There is no max pooling.
For a 320x320 input, stage resolutions are 320x320, 160x160, and 80x80.
Global average pooling produces 128 features; dropout and a linear classifier
produce the species scores. Earlier residual-model weights do not match this layout.
The config keeps the best baseline run's settings: 320-pixel inputs, 70 epochs,
batch size 32, Adam at 0.001, weight decay 0.0001, dropout 0.3, and flip/contrast augmentation.
This is a custom six-block residual CNN, not the standard ResNet-18 layout.

## Spatial attention experiment

Run `python -m src.cli train --config configs/classification_scratch_attention.yaml`.
`src/models/bird_cnn_attention.py` reuses the delayed-pooling backbone. A learned
1x1 convolution generates one spatial score map; softmax makes its weights sum
to one, and a weighted sum replaces uniform global average pooling. Attention
starts uniform and is used during both training and inference. This is a simple
spatial-attention model, not WS-DAN: no attention cropping, dropping, or part loss.
The config uses Adam 0.001, 320px inputs, batch 32, 70 epochs, and flip/contrast
augmentation, with dropout and weight decay both zero. Because regularization
also changes, comparison with the old baseline does not isolate attention alone.
For a controlled follow-up, use the same zero-dropout/zero-decay settings on the
plain delayed-pooling model as well.

## Standard ResNet-18 experiment

Run `python -m src.cli train --config configs/classification_scratch_resnet18.yaml`.
`src/models/bird_resnet18.py` explicitly defines all eight residual blocks,
using our own `ResidualBlock` class. It includes the standard 7x7 stride-2 stem,
initial max pool, four stages of two blocks, and 200 output scores.
It is randomly initialized, with no torchvision model class or pretrained download.
This is separate from the earlier six-block custom residual model.

The config uses Adam with a constant learning rate of 0.001, 90 epochs,
batch size 32, zero weight decay and zero dropout. These are experiment settings,
not a claim of an optimal published ResNet recipe. Our existing 320px resize/pad,
[0,1] tensors and flip/contrast policy are retained.
Batch normalization stays enabled. Train/validation curves and the learning rate
are logged to MLflow. No scheduler is enabled for old configs unless requested.
Both new models support the existing `analyze-validation` command and Grad-CAM.

## Augmentation experiment

Run with `--config configs/classification_scratch_augmented.yaml` for independent
50% horizontal-flip and 50% contrast-change probabilities on each training image,
after resizing/padding. Contrast strength 0.2 uses factors between 0.8 and 1.2.
Validation remains unaugmented. The baseline config disables both operations.

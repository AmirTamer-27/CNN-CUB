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
ResNet-18 supports the existing `analyze-validation` command and Grad-CAM.

## Cutout experiment

Run `python -m src.cli train --config configs/classification_scratch_delayed_pooling.yaml`.
The config matches the successful delayed-pooling run (320px, 70 epochs, Adam
0.001 constant, dropout 0.3, weight decay 0.0001, flip/contrast), adding only Cutout.
For each training image, `prepare_images` masks one random square with probability
0.5 after resize, augmentation and tensor conversion. The nominal side is sampled
from 25–30% of the input side: 80–96px at 320px. The square has a uniformly sampled
center and is clipped at image edges, so edge patches can be smaller. Fill is gray
128/255, matching padding. No bounding boxes or bird-overlap checks are used.
Validation never uses Cutout. Old configs default to Cutout off; workers stay at 2.
Cutout settings are logged to MLflow. The learning-rate schedule is unchanged.

## Validation-plateau learning-rate experiment

Run `python -m src.cli train --config configs/classification_scratch_delayed_pooling_lr_plateau.yaml`.
This copies the Cutout experiment config but disables Cutout (`cutout_probability: 0.0`)
and enables `lr_scheduler: plateau`. The Cutout config keeps `lr_scheduler: none`.
`ReduceLROnPlateau` monitors validation loss after each complete validation epoch;
it does not monitor training loss or calculate gradients from validation images.
`lr_plateau_factor: 0.5` halves the rate after more than
`lr_plateau_patience: 5` consecutive epochs without a qualifying improvement.
That means a reduction after the sixth non-improving epoch. An improvement must
beat the running best loss by more than `lr_plateau_threshold: 0.001` (absolute).
`lr_plateau_min_lr: 0.000001` sets the minimum rate. Improvements reset the counter;
the scheduler never raises the rate or stops training. Settings and per-epoch
learning rates are logged to MLflow. All these controls are editable in YAML.

## Augmentation experiment

Run with `--config configs/classification_scratch_augmented.yaml` for independent
50% horizontal-flip and 50% contrast-change probabilities on each training image,
after resizing/padding. Contrast strength 0.2 uses factors between 0.8 and 1.2.
Validation remains unaugmented. The baseline config disables both operations.

## Combined-loss experiment

Run `python -m src.cli train --config configs/classification_scratch_delayed_pooling_combined_loss.yaml`.
This uses the best Cutout + plateau-scheduler settings for 70 epochs: 320px,
Adam 0.001, weight decay 0.0001, dropout 0.3, batch size 32 and the same seeded split.
`loss: combined` enables cross-entropy + `cosine_weight` times cosine loss on
the same 200 logits, without adding another classifier. The initial weight 1.0
is an experimental choice, not the published two-classifier method.
`src/training/losses.py` defines the objective. One backward pass updates the model.

For comparable charts, `train_loss` and `val_loss` still mean cross-entropy;
the scheduler continues to monitor validation cross-entropy. Combined runs also
print and log `train_cosine_loss`, `val_cosine_loss`, `train_total_loss`, and
`val_total_loss` to MLflow. No JSON files are written. Existing configs default
to cross-entropy alone. Setting `cosine_weight: 0.0` also gives CE-only training.

## Bounding-box crop experiment

Place the original CUB annotation files at `data/CUB_200_2011/images.txt`
and `data/CUB_200_2011/bounding_boxes.txt` (exact filenames). No image download
is needed: images still come from Hugging Face. This data directory is ignored
by Git, so copy the two files into the same location in Kaggle/Colab too.

Run `python -m src.cli train --config configs/classification_scratch_delayed_pooling_bbox.yaml`.
This starts a 70-epoch experiment using the best CE + Cutout + plateau settings,
adding only bird crops. `use_bounding_boxes: false` disables cropping; old configs
default to false. `bbox_annotation_dir` is relative to the project root when
running commands there, or can be an absolute path.

`src/data/bounding_boxes.py` joins original image IDs to box annotations, then
matches Hugging Face samples by filename. It rejects missing boxes or ambiguous
matches rather than dropping samples. `dataset.py` crops the dataset once before
the unchanged seeded split, preserving row order and labels. Both training and
validation therefore use crops of the same photographs as before. There is no
margin. Fractional boundaries round outward to pixels and clip to image edges.
Existing resize/pad still scales each crop up or down to the configured size.
Cropping is currently connected to scratch-model loaders; ResNeXt uses its separate
transfer loaders. Use the matching crop config for saved-weight validation analysis.
MLflow records whether cropping is enabled and the annotation directory.

## Bounding boxes plus random cropping

Run `python -m src.cli train --config configs/best_boundingBoxesTrial_randomCrop.yaml`.
This copies `best_boundingBoxesTrial.yaml` (including 70 epochs) and adds only
training-time random cropping. `random_crop_probability: 1.0` samples a fresh
crop for every training image access; set it to 0.0 to disable it.
`random_crop_scale: [0.6, 1.0]` controls retained area relative to the bbox crop,
not width/height fractions. Crops preserve the source aspect ratio up to pixel
rounding, then go through the existing LANCZOS resize and gray padding.
The order is bbox crop -> random crop -> resize/pad -> flip/contrast -> tensor -> Cutout.
Validation uses the entire bbox crop with deterministic resize/pad, without random
cropping. No bbox margin is added. The two DataLoader workers and all other
settings stay unchanged. Old configs default to random cropping off. These
settings are logged to MLflow. This option applies to scratch-model data loaders.

## Cosine learning-rate experiment

Run `python -m src.cli train --config configs/best_boundingBoxesTrial_randomCrop_cosine.yaml`.
This copies the bbox + random-crop trial and changes only the scheduler.
`lr_scheduler: cosine` selects PyTorch `CosineAnnealingLR`, with
`lr_cosine_t_max: 70` and `lr_cosine_min_lr: 0.000001`. Adam starts at 0.001.
The scheduler steps once after each epoch without reading validation loss.
There are no restarts: LR falls smoothly over the configured duration.
Epoch 1 uses 0.001; the floor is reached after epoch 70's update, so the
last training epoch uses a rate slightly above the floor. MLflow logs the
rate actually used in each epoch and both cosine settings. Set T_max at least
as large as the epoch budget; the CLI rejects shorter durations to avoid an
LR upswing. Existing plateau/step config fields are inactive in cosine mode.
All old scheduler modes, model settings, augmentation, and two workers are retained.

## Named bounding-box baselines and convolutional dropout

`configs/bestBounding.yaml` preserves the 70-epoch Adam bbox baseline.
`configs/bestBounding_RandomCrop.yaml` preserves its 70-epoch random-crop variant.
Both explicitly set `dropout2d: 0.0`; the former disables random cropping.
`configs/bestBounding_RandomCrop_Dropout2D.yaml` uses the same Adam, weight decay
0.0001, augmentation and plateau settings, adds `dropout2d: 0.1` and runs 120 epochs.
Its run name ends in `_120`. Earlier config files are retained.

For the delayed-pooling model, Dropout2d is applied after each of the six complete
blocks (after max pooling where present). It zeros entire per-image feature maps
during training and is disabled by model.eval(). It adds no parameters or changes
to map sizes; existing ordinary dropout 0.3 before the FC remains. Old configs
default to convolutional dropout off. CLI construction and saved-weight analysis
read this setting, and MLflow logs it. No other model enables this option.


## Two-GPU grid search

From the project root on Kaggle with T4 x2 and the existing MLflow credentials:

```bash
python -u -m src.cli grid-search --config configs/bestBounding_RandomCrop_Dropout2D.yaml --epochs 120 --gpus 0 1
```

The default grid crosses learning rates `[0.0005, 0.001]` with Dropout2d
`[0.05, 0.10]`: four freshly initialized trials, at most two at once.
Each child runs the existing `train` command with only one GPU visible.
Batch size stays 32 per trial; DataLoader workers stay at 2 per loader.
The base YAML is not modified. All other settings, including the seed, split,
bounding boxes, augmentation and plateau scheduler, come from that file.
Use `--dry-run` to preview without CUDA, downloads or training.
Optional `--learning-rates`, `--dropout2d-values`, `--epochs` and `--gpus`
override the grid and budget; for example `--gpus 0` runs sequentially.

`src/training/grid_search.py` queues trials onto free GPUs. Each receives a unique
MLflow run name and shared `grid_search_id` tag. Compare `best_val_accuracy` and
`best_val_loss` in MLflow; these can occur at different epochs. Saved weights
still represent the final epoch, not the best epoch. Epoch histories upload after
each trial finishes, as in ordinary training.

Generated YAML settings and separate live text logs are in
`outputs/grid_search/<search_id>/`, which Git ignores. Read a printed `.log` path
from another notebook cell to inspect progress. Final weights remain in the
project root and upload to each trial's MLflow run. No JSON results are written.
Failed trials are reported while remaining trials continue. Interrupting the
launcher terminates its active child processes. This is a fixed grid, not Optuna
and not early pruning. Four 120-epoch trials take about 5.8 hours only if concurrent
epochs maintain 87 seconds; shared CPU resources may extend this to 6–8 hours.

## Two-GPU random search

```bash
python -u -m src.cli random-search --config configs/bestBounding_RandomCrop_Dropout2D.yaml --trials 4 --epochs 120 --gpus 0 1
```

Grid search tests every combination of the listed discrete values. Random search
samples a fixed number of combinations from ranges instead. Defaults sample the
initial learning rate log-uniformly from 0.0005–0.001 (equal probability for equal
multiplicative intervals) and Dropout2d uniformly from 0.05–0.10. These ranges
match the grid endpoints, but intermediate values can be chosen. All trials are
chosen before training; sampling does not adapt to earlier results or prune trials.

Override with `--lr-range MIN MAX`, `--dropout2d-range MIN MAX`, `--trials N`,
`--search-seed N`, `--epochs N`, or `--gpus 0` for sequential execution.
`--search-seed` defaults to 42 and reproduces the sampled combinations; it does
not change the model/data seed in YAML. Add `--dry-run` to preview without training.
The shared GPU queue still runs at most one process per selected GPU, retains
all other base settings and the two workers, and gives each trial its own MLflow
run and final weights. Tags `search_method`, `search_id`, `search_trial`, and
`search_seed` identify random trials. Generated YAML and live logs go under
`outputs/random_search/<search_id>/`; no JSON results are written.
Four random trials cost approximately the same as four grid trials with the
same epoch budget. Running both commands would mean eight trials, not four.

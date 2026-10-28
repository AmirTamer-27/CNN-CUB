# From Pixels to Language

CSE 485 / CSE 488 Deep Learning final project.

The scratch classifier from the Kaggle notebook is organized into Python modules.
Transfer learning, tuning, Grad-CAM, colorization, and captioning remain to be
implemented. MLflow logging is connected in the CLI. The project brief is in the parent folder.

## Where code goes

| File or folder | Responsibility |
| --- | --- |
| `configs/classification_scratch.yaml` | Baseline experiment settings |
| `src/data/dataset.py` | Hugging Face source, fixed train/validation split, DataLoaders |
| `src/data/transforms.py` | RGB conversion, aspect-preserving resize/padding, tensors |
| `src/models/bird_cnn.py` | Your six-block CNN architecture |
| `src/evaluation/metrics.py` | Loss, accuracy, macro recall calculations |
| `src/training/trainer.py` | Training/validation loops and in-memory metrics |
| `src/cli.py` | Read configuration and assemble an experiment |
| `notebooks/` | EDA and interactive inspection; existing notebook is preserved |
| `outputs/` | Generated run results and weights; ignored by Git |
| `reports/` | Report sources and deliverables |

## Setup and run

From the `project` directory, install dependencies in your chosen environment:

```bash
python -m pip install -r requirements.txt
python -m src.cli train --config configs/classification_scratch.yaml
```

This command starts a fresh model and optimizer. Run it on Kaggle for GPU training.
The training command downloads only the Hugging Face training source, creates
an 80/20 stratified split with seed 42, and never loads the official test source.
Internet access is required to download uncached data and dependencies.

The dependency list is not yet pinned to a verified Kaggle environment. Record
working package versions after the first successful Kaggle run.

## Reading the refactor

1. `cli.py` reads YAML settings and chooses CUDA, MPS, or CPU.
2. `make_loaders` creates the same train/validation subsets from the training source.
3. `prepare_images` resizes and pads on access, converts to tensors, and returns image/label only.
4. `BirdCNN` defines the same six explicit blocks and global average pooling.
5. `fit` calls `train_one_epoch` and then `validate` each epoch.
6. Each epoch prints metrics; metric history is returned in memory.
   The CLI saves the final model state and uploads it to MLflow.

Validation uses evaluation mode and disables gradients; it does not update weights
or BatchNorm running statistics. Training metrics accumulate batch predictions
made during training, as in the original notebook. They are not a separate
post-epoch evaluation of the final weights.

The CLI uses one PyTorch seed and explicit device selection. Automatic Git metadata
recording and the unused `--task` option have been removed for simplicity.

The refactor also honors `num_classes` in the final layer and exposes
`forward_features` for later encoder reuse. Training-only horizontal flipping and contrast changes are available through
`horizontal_flip_probability` and `contrast_change_probability`. The baseline uses
0.0 for both, and validation always disables both. `contrast_strength` controls
the range of contrast factors.
No input standardization has been added. Resizing uses the original RGB/LANCZOS/gray-padding function.

## Analyze saved validation predictions

After training saves `model_weights.pth`, inspect it from the repository root:

```bash
python -m src.cli analyze-validation --config configs/classification_scratch_delayed_pooling.yaml --weights model_weights.pth --top 12
```

Use the configuration and model code version that produced the weights: the
model name, image size, label mapping, validation fraction, and split seed must
match. Loading a state dictionary checks tensor names and sizes, but cannot detect
a changed pooling stride, since pooling has no saved weights.

This command loads the state onto CPU, copies it into the selected model, and runs
only validation inference in evaluation mode without gradients or optimizer
updates. It never loads the official test source and requires no MLflow login.
It prints overall loss/accuracy/macro recall, per-class loss and recall, common
confusion pairs, and the highest-loss samples. The full tables and an image grid
are saved in one self-contained `outputs/validation_analysis/report.html`; no JSON
is created. `--top` controls the terminal previews and image count. Use
`--output-dir outputs/another_analysis` to keep reports for different models.

In Colab/Kaggle, launch the command with `!python ...`, then download the HTML
from the notebook's file browser and open it in a browser. A subprocess cannot
display a Matplotlib plot directly in the notebook. Inputs are shown using the
current unstandardized [0, 1] tensor values. Indices refer to validation-subset
positions. Class statistics with roughly six images per class are noisy; visually
inspect failures rather than assuming an automatic diagnosis of their cause.

## Training metrics

Metrics are printed and returned by `fit` as an in-memory history list. The CLI
captures that history and uploads it to MLflow after training. Training does not
create JSON result files or copy configuration files.
MLflow records parameters and uploads epoch metrics after training finishes.
The CLI saves the final model state as `model_weights.pth` and logs it to the
MLflow run under `weights/`. This file is overwritten locally by the next training
run, while MLflow retains each run's copy. This saves the final epoch, not the best
validation epoch.

Accuracy and recall are percentages. The test set is reserved for final evaluation;
a test command is not yet implemented. A fixed seed helps repeat comparisons but
does not guarantee identical results across GPU environments.

## Trying another experiment

For delayed early downsampling, run:

```bash
python -m src.cli train --config configs/classification_scratch_delayed_pooling.yaml
```

`BirdCNNDelayedPooling` in `src/models/bird_cnn_delayed_pooling.py` keeps the
baseline's six convolution blocks, but changes the first max-pooling stride from
2 to 1. At input size 224, block output sizes are 223, 111, 55, 27, 13, and 6,
instead of 112, 56, 28, 14, 7, and 3. Global average pooling still produces 512
features. The parameter count is unchanged, but larger maps increase computation
and memory use. Its separate config matches the regularized 40-epoch experiment's
training settings. Compare validation results to test whether delaying early
downsampling helps; this is a hypothesis, not a guarantee of preserving detail.

For a learning-rate experiment, copy `classification_scratch.yaml` to a new YAML,
change `run_name` and `learning_rate`, then pass that file to the same command.
Keep the split, image preparation, and other settings fixed for a controlled comparison.
Each invocation creates fresh weights; it does not continue the preceding experiment.

For a different architecture, add a model file under `src/models/`, import it in
`src/cli.py`, and extend the model selection there. A model name in YAML alone does
not implement that model. Reuse the data and training code when compatible.
For a change to the existing architecture, commit the baseline first, then edit it;
Git preserves earlier code. Do not copy the whole project for every experiment.

Record hypotheses, results, and failures in the report. Select experiments with
validation metrics rather than training accuracy or repeated test-set checks.
MLflow logs the selected configuration settings, including augmentation parameters,
and epoch metrics to the configured DagsHub server. Supply credentials through
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`, outside the repository.

## First augmentation experiment

```bash
python -m src.cli train --config configs/classification_scratch_augmented.yaml
```

This keeps the baseline's model, split, resolution, batch size, learning rate, and
20 epochs. After resizing and padding, each training image independently has a
50% chance of horizontal flipping and a 50% chance of contrast adjustment.
Contrast strength 0.2 samples a factor between 0.8 and 1.2 when adjustment occurs.
Neither, either, or both transformations can happen. One input image still yields
one tensor with the same label; the dataset size is unchanged. Validation uses
only resizing, padding, and tensor conversion.

MLflow logs both probabilities and contrast strength under run name
`scratch_baseline_augmented`. Metrics are uploaded after `fit` returns, not live.
Compare validation performance with the baseline; lower training accuracy under
augmentation alone does not mean the experiment failed. Preview transformed
training images before a full run; these probabilities and strength are starting
choices, not optimized settings.

# From Pixels to Language

CSE 485 / CSE 488 Deep Learning final project.

The scratch classifier from the Kaggle notebook is organized into Python modules.
Transfer learning, tuning, Grad-CAM, colorization, captioning, and MLflow integration
remain to be implemented. The project brief is in the parent folder.

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
   Training does not write result files or model checkpoints.

Validation uses evaluation mode and disables gradients; it does not update weights
or BatchNorm running statistics. Training metrics accumulate batch predictions
made during training, as in the original notebook. They are not a separate
post-epoch evaluation of the final weights.

The CLI uses one PyTorch seed and explicit device selection. Automatic Git metadata
recording and the unused `--task` option have been removed for simplicity.

The refactor also honors `num_classes` in the final layer and exposes
`forward_features` for later encoder reuse. No augmentation or input standardization
has been added. Resizing uses the original RGB/LANCZOS/gray-padding function.

## Experiment results

Metrics are printed and returned by `fit` as an in-memory history list. The CLI
currently does not capture that return value. No output folder, copied configuration,
JSON result files, or checkpoints are created by the training code.
MLflow logging and saving trained weights are not connected yet.

Accuracy and recall are percentages. The test set is reserved for final evaluation;
a test command is not yet implemented. A fixed seed helps repeat comparisons but
does not guarantee identical results across GPU environments.

## Trying another experiment

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
MLflow can later log the configuration and epoch metrics; it is not connected yet.

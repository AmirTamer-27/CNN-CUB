# Project structure and training flow

## Where things go

| File or folder | What belongs here |
| --- | --- |
| `configs/` | Experiment settings: image size, batch size, epochs, learning rate, and seed. |
| `src/cli.py` | Starting point: reads settings and connects the other files. |
| `src/data/dataset.py` | Loads data, creates the train/validation split, and builds DataLoaders. |
| `src/data/transforms.py` | Resizes and pads images, then converts them to tensors when accessed. |
| `src/models/bird_cnn.py` | The CNN architecture: layers and forward pass. |
| `src/training/trainer.py` | Training batches, validation, and the loop over epochs. |
| `src/evaluation/metrics.py` | Loss, accuracy, and macro recall calculations. |
| `notebooks/` | EDA, image displays, and interactive experiments. |
| `reports/` | Project report and findings. |
| `requirements.txt` | Python dependencies needed to run the project. |

## Training flow

Run from the project directory:

```bash
python -m src.cli train --config configs/classification_scratch.yaml
```

1. `cli.py` reads the YAML configuration, sets the seed, and chooses the device.
2. It calls `make_loaders` in `dataset.py` to load the training source and reserve a validation subset. The official test set is not loaded.
3. When a DataLoader requests samples, `prepare_images` resizes, pads, and converts their images to tensors. The loader groups samples into batches.
4. The CLI creates a fresh `BirdCNN`, cross-entropy loss, and Adam optimizer.
5. It calls `fit` in `trainer.py`. Each epoch calls `train_one_epoch` to update weights, then `validate` to measure performance without updates.
6. Metrics are printed and returned as an in-memory history list. The CLI currently does not capture that list. No result files or model checkpoints are saved; MLflow is not connected yet.

## Trying another experiment

- **Different settings:** copy the baseline YAML, change the relevant setting, and pass the new file to the same command.
- **Different architecture:** add a model under `src/models/` and update the model selection in `cli.py` to construct it.
- Keep the data split fixed and compare validation results. Each command starts training from fresh weights.
- Commit and push code changes to GitHub. Update the repository copy on Kaggle before running the command there; pushing does not start training automatically.

"""Start training from the project root with python -m src.cli train --config PATH."""
import argparse

import torch
from torch import nn
import yaml

from src.data.dataset import make_loaders
from src.models.bird_cnn import BirdCNN
from src.training.trainer import fit


def main():
    # 1. Read the command and configuration file.
    parser = argparse.ArgumentParser(description="Train a bird classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    train_command = commands.add_parser("train")
    train_command.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Only these choices have been implemented so far.
    if config["model"] != "bird_cnn" or config["optimizer"] != "adam":
        raise ValueError("This version supports model: bird_cnn and optimizer: adam.")

    # 2. Set the PyTorch seed and select the device.
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 3. Prepare the data.
    train_loader, validation_loader, class_names = make_loaders(config)
    if len(class_names) != config["num_classes"]:
        raise ValueError("num_classes must match the number of dataset classes.")

    # 4. Create a fresh model, loss function, and optimizer.
    model = BirdCNN(num_classes=config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 5. Start training.
    print(f"Device: {device}")
    fit(model, train_loader, validation_loader, criterion, optimizer, device, config)


if __name__ == "__main__":
    main()

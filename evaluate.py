import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../test"),
        help="Path to test folder containing class subfolders",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pth"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    if dataset.classes != config["class_names"]:
        raise ValueError(
            f"Class order mismatch:\n"
            f"Dataset: {dataset.classes}\n"
            f"Config: {config['class_names']}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(num_classes=len(config["class_names"]))

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            if config.get("tta", False):
                flipped = torch.flip(images, dims=[3])
                flipped_logits = model(flipped)
                logits = (logits + flipped_logits) / 2

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    print(f"Device: {device}")
    print(f"Test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Expected reported accuracy: {config['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
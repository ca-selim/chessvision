import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ChessSquareCNN


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    "data/processed/squares"
)

OUTPUT_DIR = Path(
    "models/square_classifier"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 128

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("=" * 70)
print("ChessVision - Square Classifier Training")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# Image transformations
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Small realistic photographic variations.
    transforms.RandomAffine(
        degrees=3,
        translate=(0.02, 0.02),
        scale=(0.97, 1.03)
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Datasets
# ============================================================

train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transform
)


print()
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

print()
print("Class mapping:")

for name, index in train_dataset.class_to_idx.items():

    print(
        f"  {index:2d} -> {name}"
    )


# ============================================================
# Verify class mappings
# ============================================================

if train_dataset.class_to_idx != val_dataset.class_to_idx:

    raise RuntimeError(
        "Train and validation class mappings differ!"
    )


class_names = [
    name
    for name, index
    in sorted(
        train_dataset.class_to_idx.items(),
        key=lambda x: x[1]
    )
]

num_classes = len(class_names)

if num_classes != 13:

    raise RuntimeError(
        f"Expected 13 classes, "
        f"found {num_classes}"
    )


# ============================================================
# Save class mapping
# ============================================================

with open(
    OUTPUT_DIR / "class_names.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=2
    )


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# Calculate class weights
#
# We use a square-root inverse-frequency weighting.
# This compensates for imbalance without giving extremely
# rare classes excessively large weights.
# ============================================================

class_counts = np.bincount(
    train_dataset.targets,
    minlength=num_classes
)

total_samples = class_counts.sum()

raw_weights = np.sqrt(
    total_samples /
    (num_classes * class_counts)
)

# Normalize so average weight is approximately 1.
raw_weights = (
    raw_weights /
    raw_weights.mean()
)

class_weights = torch.tensor(
    raw_weights,
    dtype=torch.float32,
    device=device
)


print()
print("Class weights:")

for index, name in enumerate(class_names):

    print(
        f"  {index:2d} "
        f"{name:15s} "
        f"count={class_counts[index]:6d} "
        f"weight={raw_weights[index]:.3f}"
    )


# ============================================================
# Model
# ============================================================

model = ChessSquareCNN(
    num_classes=num_classes
)

model = model.to(device)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Learning-rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# Evaluation function
# ============================================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    class_correct = np.zeros(
        num_classes,
        dtype=np.int64
    )

    class_total = np.zeros(
        num_classes,
        dtype=np.int64
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            for class_index in range(
                num_classes
            ):

                mask = (
                    labels == class_index
                )

                class_total[class_index] += (
                    mask.sum().item()
                )

                class_correct[class_index] += (
                    (
                        predictions[mask]
                        == labels[mask]
                    )
                    .sum()
                    .item()
                )

    accuracy = correct / total

    per_class_recall = np.divide(
        class_correct,
        class_total,
        out=np.zeros_like(
            class_correct,
            dtype=float
        ),
        where=class_total != 0
    )

    macro_recall = per_class_recall.mean()

    return (
        accuracy,
        macro_recall,
        per_class_recall
    )


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    train_correct = 0
    train_total = 0


    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    # --------------------------------------------------------
    # Training statistics
    # --------------------------------------------------------

    train_loss = (
        running_loss /
        train_total
    )

    train_accuracy = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    (
        val_accuracy,
        macro_recall,
        per_class_recall
    ) = evaluate(
        model,
        val_loader
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Loss:          {train_loss:.4f}"
    )

    print(
        f"  Train accuracy: {train_accuracy:.4f}"
    )

    print(
        f"  Val accuracy:   {val_accuracy:.4f}"
    )

    print(
        f"  Macro recall:   {macro_recall:.4f}"
    )

    print(
        f"  Learning rate:  {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # Per-class validation recall
    # --------------------------------------------------------

    print("\n  Validation recall:")

    for index, name in enumerate(
        class_names
    ):

        print(
            f"    {name:15s}: "
            f"{per_class_recall[index]:.4f}"
        )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        checkpoint = {
            "model_state_dict":
                model.state_dict(),

            "class_names":
                class_names,

            "image_size":
                IMAGE_SIZE,

            "best_val_accuracy":
                best_val_accuracy,

            "epoch":
                epoch,
        }

        checkpoint_path = (
            OUTPUT_DIR /
            "best_model.pth"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        print(
            f"\n  ✓ Saved best model: "
            f"{checkpoint_path}"
        )


# ============================================================
# Finish
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Model:",
    OUTPUT_DIR / "best_model.pth"
)
import csv
from collections import Counter
from pathlib import Path


METADATA_FILE = Path(
    "data/processed/squares/metadata.csv"
)


# ============================================================
# Load metadata
# ============================================================

counts = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter(),
}

total = 0

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        split = row["split"]
        class_name = row["class"]

        counts[split][class_name] += 1
        total += 1


# ============================================================
# Class order
# ============================================================

classes = [
    "empty",

    "white-pawn",
    "white-rook",
    "white-knight",
    "white-bishop",
    "white-queen",
    "white-king",

    "black-pawn",
    "black-rook",
    "black-knight",
    "black-bishop",
    "black-queen",
    "black-king",
]


# ============================================================
# Print statistics
# ============================================================

print("=" * 75)
print("ChessVision - Dataset Class Distribution")
print("=" * 75)

for split in ["train", "val", "test"]:

    print()
    print(f"[{split.upper()}]")
    print("-" * 75)

    split_total = sum(
        counts[split].values()
    )

    print(f"Total squares: {split_total}")
    print()

    print(
        f"{'Class':20s}"
        f"{'Count':>10s}"
        f"{'Percent':>12s}"
    )

    print("-" * 45)

    for class_name in classes:

        count = counts[split][class_name]

        percentage = (
            count / split_total * 100
            if split_total > 0
            else 0
        )

        print(
            f"{class_name:20s}"
            f"{count:10d}"
            f"{percentage:11.2f}%"
        )


# ============================================================
# Compare splits
# ============================================================

print()
print("=" * 75)
print("CLASS DISTRIBUTION SUMMARY")
print("=" * 75)

print(
    f"{'Class':20s}"
    f"{'Train':>10s}"
    f"{'Val':>10s}"
    f"{'Test':>10s}"
)

print("-" * 55)

for class_name in classes:

    print(
        f"{class_name:20s}"
        f"{counts['train'][class_name]:10d}"
        f"{counts['val'][class_name]:10d}"
        f"{counts['test'][class_name]:10d}"
    )


print()
print("Total samples:", total)
import random
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(
    "data/processed/squares/train"
)

OUTPUT_DIR = Path(
    "outputs/class_samples"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLES_PER_CLASS = 5

IMAGE_SIZE = 128

# Keep the class order consistent with our classifier.
CLASSES = [
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
# Text helper
# ============================================================

def add_label(image, text):
    """
    Add class name above the image.
    """

    label_height = 35

    canvas = np.full(
        (
            IMAGE_SIZE + label_height,
            IMAGE_SIZE,
            3
        ),
        255,
        dtype=np.uint8
    )

    canvas[
        label_height:,
        :
    ] = image

    cv2.putText(
        canvas,
        text,
        (5, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# Generate contact sheet for each class
# ============================================================

for class_name in CLASSES:

    class_dir = DATASET_DIR / class_name

    files = list(
        class_dir.glob("*.jpg")
    )

    if not files:
        print(
            f"WARNING: No images found for "
            f"{class_name}"
        )

        continue

    random.shuffle(files)

    selected = files[
        :SAMPLES_PER_CLASS
    ]

    print(
        f"{class_name:15s}: "
        f"{len(files):6d} available"
    )

    samples = []

    for file in selected:

        image = cv2.imread(
            str(file)
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        labeled = add_label(
            image,
            class_name
        )

        samples.append(labeled)


    if not samples:
        continue


    # --------------------------------------------------------
    # Arrange samples horizontally
    # --------------------------------------------------------

    sheet = np.hstack(samples)

    output_path = (
        OUTPUT_DIR
        / f"{class_name}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        sheet
    )


print()
print("=" * 60)
print("CLASS VISUALIZATION COMPLETE")
print("=" * 60)

print(
    "Output directory:",
    OUTPUT_DIR
)
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("data/raw")
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"

OUTPUT_DIR = Path("outputs/sample_squares")

IMAGE_ID = 0

WARP_SIZE = 1024
SQUARE_SIZE = WARP_SIZE // 8


# ============================================================
# Load annotations
# ============================================================

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


images = data["images"]
pieces = data["annotations"]["pieces"]
corners = data["annotations"]["corners"]
categories = data["categories"]


# ============================================================
# Lookup tables
# ============================================================

image_by_id = {
    image["id"]: image
    for image in images
}

category_by_id = {
    category["id"]: category["name"]
    for category in categories
}

corner_by_image_id = {
    corner["image_id"]: corner
    for corner in corners
}


# ============================================================
# Get image
# ============================================================

image_info = image_by_id[IMAGE_ID]

image_path = DATASET_DIR / image_info["path"]

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(
        f"Could not load image:\n{image_path}"
    )


# ============================================================
# Get corners
# ============================================================

corner_data = corner_by_image_id[IMAGE_ID]
corner_points = corner_data["corners"]


src_points = np.float32([
    corner_points["top_left"],
    corner_points["top_right"],
    corner_points["bottom_right"],
    corner_points["bottom_left"],
])


dst_points = np.float32([
    [0, 0],
    [WARP_SIZE - 1, 0],
    [WARP_SIZE - 1, WARP_SIZE - 1],
    [0, WARP_SIZE - 1],
])


# ============================================================
# Perspective warp
# ============================================================

matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)

warped = cv2.warpPerspective(
    image,
    matrix,
    (WARP_SIZE, WARP_SIZE)
)


# ============================================================
# Build piece lookup
#
# Example:
#
# ("a8") -> "black-rook"
# ============================================================

piece_lookup = {}

for piece in pieces:

    if piece["image_id"] != IMAGE_ID:
        continue

    square = piece["chessboard_position"]

    category_id = piece["category_id"]

    category_name = category_by_id[category_id]

    piece_lookup[square] = category_name


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Generate 64 squares
# ============================================================

print("=" * 60)
print("Generating squares for image:", IMAGE_ID)
print("=" * 60)

for row in range(8):

    for col in range(8):

        # --------------------------------------------------
        # Pixel coordinates
        # --------------------------------------------------

        x1 = col * SQUARE_SIZE
        y1 = row * SQUARE_SIZE

        x2 = (col + 1) * SQUARE_SIZE
        y2 = (row + 1) * SQUARE_SIZE


        # --------------------------------------------------
        # Crop square
        # --------------------------------------------------

        crop = warped[
            y1:y2,
            x1:x2
        ]


        # --------------------------------------------------
        # Algebraic coordinate
        # --------------------------------------------------

        file_name = chr(ord("a") + col)
        rank = 8 - row

        chess_square = f"{file_name}{rank}"


        # --------------------------------------------------
        # Determine class
        # --------------------------------------------------

        category_name = piece_lookup.get(
            chess_square,
            "empty"
        )


        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        filename = (
            f"image_{IMAGE_ID:05d}_"
            f"{chess_square}_"
            f"{category_name}.jpg"
        )

        output_path = OUTPUT_DIR / filename

        cv2.imwrite(
            str(output_path),
            crop
        )


        print(
            f"{chess_square:3s} -> "
            f"{category_name:13s} -> "
            f"{output_path.name}"
        )


print("\nGenerated 64 square images.")
print("Output:", OUTPUT_DIR)
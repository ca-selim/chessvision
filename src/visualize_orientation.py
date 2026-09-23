import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("data/raw")
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"

OUTPUT_DIR = Path("outputs/orientation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We'll start with the first test image.
IMAGE_ID = 0

# Size of the corrected chessboard.
WARP_SIZE = 1024


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
# Build lookup tables
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


pieces_by_image_id = {}

for piece in pieces:
    image_id = piece["image_id"]

    if image_id not in pieces_by_image_id:
        pieces_by_image_id[image_id] = []

    pieces_by_image_id[image_id].append(piece)


# ============================================================
# Get image information
# ============================================================

image_info = image_by_id[IMAGE_ID]

image_path = DATASET_DIR / image_info["path"]

print("Image:")
print(image_path)

print("Exists:", image_path.exists())

if not image_path.exists():
    raise FileNotFoundError(
        f"Could not find image:\n{image_path}"
    )


# ============================================================
# Load image
# ============================================================

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(
        f"OpenCV could not read:\n{image_path}"
    )

print("Image shape:", image.shape)


# ============================================================
# Get corners
# ============================================================

corner_data = corner_by_image_id[IMAGE_ID]

corner_points = corner_data["corners"]

print("\nCorners:")
for name, point in corner_points.items():
    print(f"  {name}: {point}")


# ============================================================
# Convert corners to OpenCV format
#
# Required order:
#
#   top-left
#   top-right
#   bottom-right
#   bottom-left
# ============================================================

src_points = np.float32([
    corner_points["top_left"],
    corner_points["top_right"],
    corner_points["bottom_right"],
    corner_points["bottom_left"],
])


# ============================================================
# Destination coordinates
# ============================================================

dst_points = np.float32([
    [0, 0],
    [WARP_SIZE - 1, 0],
    [WARP_SIZE - 1, WARP_SIZE - 1],
    [0, WARP_SIZE - 1],
])


# ============================================================
# Perspective transformation
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
# Draw corners on original image
# ============================================================

original_visual = image.copy()

corner_colors = {
    "top_left": (255, 0, 0),
    "top_right": (0, 255, 0),
    "bottom_right": (0, 0, 255),
    "bottom_left": (255, 255, 0),
}

for name, point in corner_points.items():

    x, y = map(int, point)

    cv2.circle(
        original_visual,
        (x, y),
        20,
        corner_colors[name],
        -1
    )

    cv2.putText(
        original_visual,
        name,
        (x + 25, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        corner_colors[name],
        3,
        cv2.LINE_AA,
    )


# ============================================================
# Draw 8x8 grid
# ============================================================

square_size = WARP_SIZE // 8

for row in range(8):

    for col in range(8):

        x1 = col * square_size
        y1 = row * square_size

        x2 = (col + 1) * square_size
        y2 = (row + 1) * square_size

        # Grid
        cv2.rectangle(
            warped,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        # Algebraic coordinate
        file_name = chr(ord("a") + col)
        rank = 8 - row

        square = f"{file_name}{rank}"

        cv2.putText(
            warped,
            square,
            (x1 + 8, y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ============================================================
# Transform piece bounding-box centers
#
# This is ONLY for visualization.
# We are NOT going to use the bounding boxes
# as our classifier input.
# ============================================================

piece_visual = warped.copy()

for piece in pieces_by_image_id.get(IMAGE_ID, []):

    category_name = category_by_id[
        piece["category_id"]
    ]

    chess_square = piece["chessboard_position"]

    bbox = piece.get("bbox")

    if bbox is None:
        continue

    x, y, width, height = bbox

    center_x = x + width / 2
    center_y = y + height / 2

    point = np.float32([
        [[center_x, center_y]]
    ])

    transformed = cv2.perspectiveTransform(
        point,
        matrix
    )

    warped_x, warped_y = transformed[0][0]

    warped_x = int(warped_x)
    warped_y = int(warped_y)

    # Mark the detected piece center.
    cv2.circle(
        piece_visual,
        (warped_x, warped_y),
        8,
        (0, 0, 255),
        -1,
    )
    
    label = f"{chess_square}: {category_name}"

    # Determine which square this piece belongs to.
    col = ord(chess_square[0]) - ord("a")
    row = 8 - int(chess_square[1])

    square_x = col * square_size
    square_y = row * square_size

    # Put the label safely inside the square.
    text_x = square_x + 8
    text_y = square_y + 55

    cv2.putText(
        piece_visual,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

# ============================================================
# Save results
# ============================================================

original_output = OUTPUT_DIR / f"original_{IMAGE_ID}.jpg"
warped_output = OUTPUT_DIR / f"warped_{IMAGE_ID}.jpg"
piece_output = OUTPUT_DIR / f"warped_pieces_{IMAGE_ID}.jpg"

cv2.imwrite(
    str(original_output),
    original_visual
)

cv2.imwrite(
    str(warped_output),
    warped
)

cv2.imwrite(
    str(piece_output),
    piece_visual
)


# ============================================================
# Done
# ============================================================

print("\nSaved:")
print(original_output)
print(warped_output)
print(piece_output)

print("\nOpen this image first:")
print(piece_output)
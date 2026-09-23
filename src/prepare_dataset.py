import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("data/raw")
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"

OUTPUT_DIR = Path("data/processed/squares")

WARP_SIZE = 1024
SQUARE_SIZE = WARP_SIZE // 8


# ============================================================
# Load annotations
# ============================================================

print("Loading annotations...")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


images = data["images"]
pieces = data["annotations"]["pieces"]
corners = data["annotations"]["corners"]
categories = data["categories"]
splits = data["splits"]


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
# Build piece lookup
#
# piece_lookup[image_id][square] = category
#
# Example:
#
# piece_lookup[0]["a8"] = "black-rook"
# ============================================================

print("Building piece lookup...")

piece_lookup = {}

for piece in pieces:

    image_id = piece["image_id"]
    square = piece["chessboard_position"]

    category_name = category_by_id[
        piece["category_id"]
    ]

    if image_id not in piece_lookup:
        piece_lookup[image_id] = {}

    piece_lookup[image_id][square] = category_name


# ============================================================
# Prepare output directories
# ============================================================

for split_name in ["train", "val", "test"]:

    split_dir = OUTPUT_DIR / split_name

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Process one image
# ============================================================

def process_image(image_id, split_name, writer):

    image_info = image_by_id[image_id]

    image_path = DATASET_DIR / image_info["path"]

    if not image_path.exists():
        print(f"WARNING: Missing image: {image_path}")
        return 0

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read: {image_path}")
        return 0


    # --------------------------------------------------------
    # Get corners
    # --------------------------------------------------------

    if image_id not in corner_by_image_id:

        print(
            f"WARNING: No corners for image {image_id}"
        )

        return 0

    corner_data = corner_by_image_id[image_id]

    corners_data = corner_data["corners"]


    # --------------------------------------------------------
    # Corner order:
    #
    # top-left
    # top-right
    # bottom-right
    # bottom-left
    # --------------------------------------------------------

    src_points = np.float32([
        corners_data["top_left"],
        corners_data["top_right"],
        corners_data["bottom_right"],
        corners_data["bottom_left"],
    ])


    dst_points = np.float32([
        [0, 0],
        [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1],
        [0, WARP_SIZE - 1],
    ])


    # --------------------------------------------------------
    # Perspective transformation
    # --------------------------------------------------------

    matrix = cv2.getPerspectiveTransform(
        src_points,
        dst_points
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (WARP_SIZE, WARP_SIZE)
    )


    # --------------------------------------------------------
    # Get piece labels for this image
    # --------------------------------------------------------

    image_pieces = piece_lookup.get(
        image_id,
        {}
    )


    # --------------------------------------------------------
    # Generate 64 squares
    # --------------------------------------------------------

    count = 0

    for row in range(8):

        for col in range(8):

            # Pixel coordinates
            x1 = col * SQUARE_SIZE
            y1 = row * SQUARE_SIZE

            x2 = (col + 1) * SQUARE_SIZE
            y2 = (row + 1) * SQUARE_SIZE


            # Crop
            crop = warped[
                y1:y2,
                x1:x2
            ]


            # Algebraic square
            file_name = chr(
                ord("a") + col
            )

            rank = 8 - row

            chess_square = (
                f"{file_name}{rank}"
            )


            # ------------------------------------------------
            # Determine class
            #
            # No piece annotation = empty
            # ------------------------------------------------

            category_name = image_pieces.get(
                chess_square,
                "empty"
            )


            # ------------------------------------------------
            # Output directory
            # ------------------------------------------------

            class_dir = (
                OUTPUT_DIR
                / split_name
                / category_name
            )

            class_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            # ------------------------------------------------
            # Filename
            # ------------------------------------------------

            filename = (
                f"image_{image_id:05d}_"
                f"{chess_square}.jpg"
            )

            output_path = class_dir / filename


            # ------------------------------------------------
            # Save crop
            # ------------------------------------------------

            success = cv2.imwrite(
                str(output_path),
                crop
            )

            if not success:

                print(
                    f"WARNING: Failed to save "
                    f"{output_path}"
                )

                continue


            # ------------------------------------------------
            # Metadata CSV
            # ------------------------------------------------

            writer.writerow([
                split_name,
                image_id,
                image_info["file_name"],
                image_info["game_id"],
                image_info["move_id"],
                chess_square,
                category_name,
                str(output_path),
            ])

            count += 1


    return count


# ============================================================
# Main processing
# ============================================================

metadata_path = OUTPUT_DIR / "metadata.csv"

total_processed = 0

print()
print("=" * 70)
print("GENERATING CHESSRED2K SQUARE DATASET")
print("=" * 70)


with open(
    metadata_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "split",
        "image_id",
        "file_name",
        "game_id",
        "move_id",
        "chess_square",
        "class",
        "crop_path",
    ])


    # --------------------------------------------------------
    # Process official ChessReD2K splits
    # --------------------------------------------------------

    for split_name in ["train", "val", "test"]:

        image_ids = splits[
            "chessred2k"
        ][split_name]["image_ids"]

        print()
        print(
            f"Processing {split_name}: "
            f"{len(image_ids)} boards"
        )

        split_total = 0

        for index, image_id in enumerate(
            image_ids,
            start=1
        ):

            count = process_image(
                image_id,
                split_name,
                writer
            )

            split_total += count

            # Progress every 50 boards
            if index % 50 == 0 or index == len(image_ids):

                print(
                    f"  {index}/{len(image_ids)} "
                    f"boards | "
                    f"{split_total} squares"
                )

        print(
            f"Completed {split_name}: "
            f"{split_total} squares"
        )

        total_processed += split_total


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("DATASET GENERATION COMPLETE")
print("=" * 70)

print(
    f"Total square images: {total_processed}"
)

print(
    f"Metadata: {metadata_path}"
)

print(
    f"Dataset directory: {OUTPUT_DIR}"
)
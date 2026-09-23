import torch
import torch.nn as nn


class ChessSquareCNN(nn.Module):
    """
    Lightweight CNN for classifying a single
    perspective-corrected chessboard square.

    Input:
        3 x 128 x 128 RGB image

    Output:
        13 class logits
    """

    def __init__(self, num_classes=13):
        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------
            nn.Conv2d(
                128,
                192,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # ------------------------------------------------
            # Global Average Pooling
            # ------------------------------------------------
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(
                192,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x
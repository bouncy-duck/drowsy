import torch
import torch.nn as nn


class DrowsyCNN(nn.Module):
    """
    Input : (B, T, F)  with T = WINDOW_SIZE, F = len(FEATURE_COLS)
    Output: (B, NUM_CLASSES) logits
    """

    def __init__(self, in_features: int = 2, num_classes: int = 3,
                 channels=(32, 64, 128), dropout: float = 0.3):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv1d(in_features, c1, kernel_size=5, padding=2),
            nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)                       # (B, F, T)
        x = self.features(x).squeeze(-1)            # (B, C)
        return self.classifier(x)                   # (B, K)
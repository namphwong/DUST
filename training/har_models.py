"""Models shared by local training, Kaggle, evaluation and later export."""
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.SwinTransformer_cls import SwinTransformer_1D


class CNNBaseline(nn.Module):
    def __init__(self, channels=(32, 64, 128), dropout=0.1):
        super().__init__()
        blocks = []
        previous = 3
        for width in channels:
            blocks.extend([nn.Conv1d(previous, width, 5, padding=2), nn.BatchNorm1d(width),
                           nn.ReLU(), nn.MaxPool1d(2)])
            previous = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(previous, 6))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(config):
    if config['model'] == 'swin':
        return SwinTransformer_1D(**config['model_kwargs'])
    if config['model'] == 'cnn':
        return CNNBaseline(**config['model_kwargs'])
    raise ValueError('Unknown model: ' + config['model'])

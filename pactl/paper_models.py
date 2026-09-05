"""Architectures used by the output-space PAC--Bayes comparison paper.

These models are intentionally independent of timm so the benchmark runner can
use the PyTorch stack preinstalled in Kaggle notebooks.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PaperMNISTCNN(nn.Module):
    """Two-block MNIST CNN with the paper's 32-dimensional feature map."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.feature_projection = nn.Linear(64 * 7 * 7, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = torch.flatten(x, 1)
        return torch.tanh(self.feature_projection(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class PreActWideBasic(nn.Module):
    """Pre-activation residual block used by WRN-28-4."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride,
                bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preactivated = F.relu(self.bn1(x), inplace=False)
        residual = x if self.shortcut is None else self.shortcut(preactivated)
        out = self.conv1(preactivated)
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        return residual + out


class PaperWideResNet(nn.Module):
    """Pre-activation WRN-28-4 with the paper's projected feature head."""

    def __init__(
        self,
        num_classes: int,
        depth: int = 28,
        widen_factor: int = 4,
        feature_dim: int = 128,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("Wide ResNet depth must satisfy (depth - 4) % 6 == 0")
        blocks_per_group = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.stem = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(
            widths[0], widths[1], blocks_per_group, stride=1
        )
        self.group2 = self._make_group(
            widths[1], widths[2], blocks_per_group, stride=2
        )
        self.group3 = self._make_group(
            widths[2], widths[3], blocks_per_group, stride=2
        )
        self.final_bn = nn.BatchNorm2d(widths[3])
        self.feature_projection = nn.Linear(widths[3], feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._initialize()

    @staticmethod
    def _make_group(
        in_channels: int,
        out_channels: int,
        block_count: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [PreActWideBasic(in_channels, out_channels, stride=stride)]
        blocks.extend(
            PreActWideBasic(out_channels, out_channels)
            for _ in range(1, block_count)
        )
        return nn.Sequential(*blocks)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.final_bn(x), inplace=False)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        x = self.feature_projection(x)
        x = self.feature_norm(x)
        return torch.tanh(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def create_paper_model(dataset: str) -> nn.Module:
    if dataset == "mnist":
        return PaperMNISTCNN(num_classes=10, feature_dim=32)
    if dataset == "cifar10":
        return PaperWideResNet(num_classes=10, feature_dim=128)
    if dataset == "cifar100":
        return PaperWideResNet(num_classes=100, feature_dim=256)
    raise ValueError(f"Unsupported dataset: {dataset}")

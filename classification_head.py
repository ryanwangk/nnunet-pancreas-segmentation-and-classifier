import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Classification head for nnUNet encoder output.
    Input: encoded feature map (B, C, D, H, W)
    Output: logits for 3 subtypes
    """
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
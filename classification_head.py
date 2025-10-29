import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Classification head with dropout regularization.
    Takes encoder bottleneck features and outputs class logits.
    """

    def __init__(self, in_channels, num_classes, dropout_p=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
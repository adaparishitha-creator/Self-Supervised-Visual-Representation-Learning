import torch.nn as nn
from torchvision import models


def build_model(num_classes=16):
    """
    Build the ResNet-18 classifier used in Experiment 5.
    No pretrained weights are downloaded.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

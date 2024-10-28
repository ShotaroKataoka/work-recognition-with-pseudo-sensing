import torch
import torch.nn as nn
import torchvision

def get_model(num_classes=12, pretrained=True):
    model = torchvision.models.video.r3d_18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

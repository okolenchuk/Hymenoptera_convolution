import torch
from Resnet18_model import *
from VGG16_model import *


# ResNet18 custom
def resnet18custom(n_classes):
    return Resnet18Custom(n_classes)


# VGG16 custom
def vgg16custom(n_classes):
    return VGG16Custom(n_classes)


# ResNet18 finetune
def resnet18(num_classes=2, pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model


# VGG16 finetune
def vgg16(num_classes=2, pretrained=True):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

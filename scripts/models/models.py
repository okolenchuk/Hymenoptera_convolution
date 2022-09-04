import torch
import torchvision
from scripts.models.Resnet18 import *
from scripts.models.VGG16_model import *


# ResNet18 custom
def resnet18custom(n_classes):
    return Resnet18Custom(ResidualBlock, [3, 4, 6, 3], n_classes)


# VGG16 custom
def vgg16custom(n_classes):
    return VGG16Custom(n_classes)


# ResNet18 finetune
def resnet18(num_classes=2, pretrained=True):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model


# VGG16 finetune
def vgg16(num_classes=2, pretrained=True):
    model = torchvision.models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

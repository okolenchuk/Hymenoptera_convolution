import torch
from Resnet18_models import *


# ResNet18 custom
def ResNet18_cus(n_classes):
    return Resnet18Custom(n_classes)


# VGG16 custom
def VGG16Custom():
    return


# ResNet18 finetune
def Resnet18(num_classes=2, pretrained=True):
    model = torch.hub.load('pytorch/Resnet18', 'resnet18', pretrained=pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model


# VGG16 finetune
def VGG16(num_classes=2, pretrained=True):
    model = torch.hub.load('pytorch/VGG16', 'vgg16', pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

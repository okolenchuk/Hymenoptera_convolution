import os
# from tqdm.autonotebook import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

use_gpu = torch.cuda.is_available()
device = torch.device("cuda")


def data_transform (dataset_path):
    r"""Преобразование обучающих данных для расширения обучающей выборки и её нормализация
        применяем Crop, Horizontal flip и Normalize
        Для валидационной (тестовой) выборки только нормализация
        Функция возвращает test и train dataloader """
    transform_pipeline = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(244),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(dataset_path, transform_pipeline[x]) for x in ['train', 'val']}
    # специальный класс для загрузки данных в виде батчей
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=2)
                   for x in ['train', 'val']}

    return dataloaders['train'], dataloaders['val']
    # папка с данными. Если запускаете в колабе, нужно скопировать данные к себе в директорию и примонтировать диск. Если запускаете локально -- просто скачайте данные
data_dir = './hymenoptera_data'
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes


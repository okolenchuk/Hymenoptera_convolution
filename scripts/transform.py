from torchvision import datasets, transforms
import os
import torch


def data_transform(dataset_path, batch_size=16, use_transform=False):
    r"""Преобразование обучающих данных для расширения обучающей выборки и её нормализация
        применяем Crop, Horizontal flip и Normalize
        Для валидационной (тестовой) выборки только нормализация
        Функция возвращает test и train dataloader """
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(244),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            normalize
        ]),
    }

    if use_transform:
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), transform[x]) for x in
                          ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True, num_workers=2) for x in ['train', 'val']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), normalize) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True, num_workers=2) for x in ['train', 'val']}

    return dataloaders

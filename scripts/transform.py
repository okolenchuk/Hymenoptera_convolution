from torchvision import datasets, transforms
import os
import torch

def data_transform (dataset_path, batch_size=16):
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
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), transform_pipeline[x]) for x in ['train', 'val']}
    # специальный класс для загрузки данных в виде батчей
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
                   for x in ['train', 'val']}

    return dataloaders['train'], dataloaders['val']
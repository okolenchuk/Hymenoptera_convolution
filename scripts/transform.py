from torchvision import datasets, transforms
import os
import torch


def data_transform(dataset_path, batch_size=16, use_transform=True):
    normalize = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(244),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(244),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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

    if use_transform:
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), transform[x]) for x in
                          ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True) for x in ['train', 'val']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x), normalize) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print('\n'.join([f'Create {x} dataloader with {len(dataloaders[x])} batches' for x in dataloaders]))
    print(f"Class names are: {', '.join(class_names)}")
    return dataloaders, dataset_sizes, class_names

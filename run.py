from scripts.transform import *
from scripts.train import *
import torch.optim as optim
from scripts.models.init_models import *
import argparse

use_gpu = torch.cuda.is_available()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments...')
    parser.add_argument('--dataset', required=True, help='Enter dataset path')
    parser.add_argument('--batch_size', type=int, default=32, help='Enter batch size')
    parser.add_argument('--use_transform', action="store_true", required=True, default=True,
                        help='If you want to add augmentations to your dataset')
    parser.add_argument('--use_model', type=str, required=True,
                        choices=['ResNet_custom', 'VGG16_custom', 'Resnet18', 'VGG16'], help='Choose model')
    parser.add_argument('--pretrained', type=bool, action="store_true", required=True, default=True,
                        help='Use pretrained model or not')
    parser.add_argument('--save_to', required=True, help='Enter path to save weights')

    args = parser.parse_args()
    path, batch_size, use_transform, model = args.input, args.batch_size, args.use_transform, args.use_model
    pretrained, save_path = args.pretrained, args.save_to
dataloaders, dataset_sizes, num_classes = data_transform(path, batch_size=batch_size, use_transform=use_transform)

# Now train the model

loss_fn = nn.CrossEntropyLoss()

if model == 'ResNet_custom':
    resnet18_cus = resnet18custom(n_classes=num_classes)
    optimizer_ft = optim.Adam(resnet18_cus.parameters(), lr=1e-3)
    resnet18_cus, losses = train_model(resnet18_cus, dataloaders, dataset_sizes, loss_fn, optimizer_ft, num_epochs=20,
                                       use_gpu=use_gpu, PATH=save_path)
elif model == 'VGG16_custom':
    vgg16_cus = vgg16custom(n_classes=num_classes)
    optimizer_ft = optim.Adam(vgg16_cus.parameters(), lr=1e-3)
    vgg16_cus, losses = train_model(vgg16_cus, dataloaders, dataset_sizes, loss_fn, optimizer_ft, num_epochs=20,
                                    use_gpu=use_gpu, PATH=save_path)
elif model == 'Resnet18':
    resnet18_torch = resnet18(num_classes=2, pretrained=pretrained)
    optimizer_ft = optim.Adam(resnet18_torch.parameters(), lr=1e-3)
    resnet18_cus, losses = train_model(resnet18_torch, dataloaders, dataset_sizes, loss_fn, optimizer_ft, num_epochs=20,
                                       use_gpu=use_gpu, PATH=save_path)
elif model == 'VGG16':
    vgg16_torch = vgg16(num_classes=2, pretrained=pretrained)
    optimizer_ft = optim.Adam(vgg16_torch.parameters(), lr=1e-3)
    resnet18_cus, losses = train_model(vgg16_torch, dataloaders, dataset_sizes, loss_fn, optimizer_ft, num_epochs=20,
                                       use_gpu=use_gpu, PATH=save_path)



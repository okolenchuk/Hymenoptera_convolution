from scripts.transform import *
from scripts.train import *
import torch.optim as optim
from scripts.models.models import *
import argparse

use_gpu = torch.cuda.is_available()


# path = r'D:\EDUCATION\IT_academy\Test_models\Datasets\hymenoptera_data\hymenoptera_data'
# path = input('Еnter the path to the dataset')
#
#
# print()
#
# resnet18_cus = resnet18custom(n_classes=2)
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer_ft = optim.Adam(resnet18_cus.parameters(), lr=1e-3)
#
# resnet18_cus, losses = train_model(resnet18_cus, dataloaders, dataset_sizes, loss_fn, optimizer_ft, num_epochs=20,
#                                    use_gpu=use_gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments...')
    parser.add_argument('--dataset', required=True, help='Enter dataset path')
    parser.add_argument('--batch_size', type='int', required=True, help='Enter batch size')
    parser.add_argument('--datatransform', type='bool', required=True, default=True,
                        help='If you want to add augmentations to your dataset')

    args = parser.parse_args()
    path, batch_size, use_transform = args.input, args.batch_size, args.datatransform

dataloaders, dataset_sizes = data_transform(path, batch_size=batch_size, use_transform=use_transform)




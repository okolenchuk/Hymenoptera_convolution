from scripts.transform import *
from scripts.train import *
import torch.optim as optim
from scripts.models.models import *

import numpy as np
import torchvision


use_gpu = torch.cuda.is_available()


# path = r'D:\EDUCATION\IT_academy\Test_models\Hymenoptera_convolution\hymenoptera_data\hymenoptera_data'
path = input('Еnter the path to the dataset')
dataloaders, dataset_sizes = data_transform(path, batch_size=8)
print('\n'.join([f'{x} dataloader with {len(dataloaders[x])} batches' for x in dataloaders]))
print()

model = resnet18custom(n_classes=2)

loss_fn = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

resnet18_cus, losses = train_model(model, dataloaders, dataset_sizes, loss_fn, optimizer_ft,  num_epochs=1, use_gpu=use_gpu)







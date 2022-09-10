# Hymenoptera_convolution
Resnet on Hymenoptera dataset  https://download.pytorch.org/tutorial/hymenoptera_data.zip

Script parameters for run.py:
1. --dataset - Path to dataset on computer that contains the folders train and eval with data
2. --batch_size - How many inputs to one batch
3. --num_epoch - Number of epochs to train model
4. --use_transform - Use if you want to add augmentations to dataset. Used: RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip.
5. --use_model - Which model to use for training. Choose one of 
      - ResNet_custom (custom realization of Resnet18 in scripts/models/Resnet18.py) 
      - VGG16_custom (custom realization of VGG16 scripts/models/VGG16_model.py) 
      - Resnet18 (from torchvision.models)
      - VGG16 (from torchvision.models)
6. --pretrained - Use pretrained model or not, default is False
7. --save_to - Path to save weights of trained model, default is .scripts/models/weights, will be created if not exist.

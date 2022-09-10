import time
import torch
import os


def train_model(model, dataloaders, dataset_sizes, criterion,
                optimizer, num_epochs=20, use_gpu=False, PATH='scripts/models/weights'):
    since = time.time()
    print(f'\nStart training the model {model.__class__.__name__}...')

    best_model_wts = model.state_dict()
    best_acc = 0.0

    losses = {'train': [], "val": []}

    if use_gpu:
        model.cuda()

    for epoch in range(num_epochs):

        print(f'\nStart {epoch+1} epoch of training')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = inputs, labels

                if phase == "train":
                    optimizer.zero_grad()

                # forward pass
                if phase == "eval":
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                preds = torch.argmax(outputs, -1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += int(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            losses[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    torch.save(best_model_wts, PATH+f'/{model.__class__.__name__}.pth')
    return model, losses

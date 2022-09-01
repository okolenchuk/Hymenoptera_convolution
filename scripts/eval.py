import time

def eval(model, dataloader, loss, use_gpu=False):
    model.eval()

    for X, y in dataloader:
        if use_gpu == True:
            X = X.cuda()
            y = y.cuda()

        y_pred = model(X)

        l = loss(y_pred, y)

        test_loss += l.item()

        test_acc += (y_pred.argmax(dim=1) == y).sum().item()
        test_iters += 1

        test_passed += len(X)

    print("ep: {}, taked: {:.3f}, train_loss: {}, train_acc: {}, test_loss: {}, test_acc: {}".format(
        ep, time.time() - start, train_loss / train_iters, train_acc / train_passed,
            test_loss / test_iters, test_acc / test_passed)
    )


def train_model(model, dataloader, criterion, optimizer, num_epochs=20, use_gpu=False):
    since = time.time()

    for epoch in num_epochs:
        model.train(True)  # установаить модель в режим обучения
        running_loss = 0.0
        running_corrects = 0

    # итерируемся по батчам
        for data in dataloader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs, labels = inputs, labels

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += (outputs.argmax(axis=1) == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects / len(dataloader)

        print('{} epoch. Train Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    best_model_wts = model.state_dict()



import torch

import torch.nn as nn
import torch.optim as optim

import numpy as np

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import copy

from ..resnet_cifar import resnet32
from ..Cifar100.utils import Cifar100

DEVICE = 'cuda'
NUM_CLASSES = 10
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1
CLASSES_BATCH = 10
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
LR = 2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001
NUM_EPOCHS = 70


def test(net, test_dataloader, n_net):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    net.to(DEVICE)
    net.train(False)

    running_loss = 0.0
    running_corrects = 0
    for index, images, labels in test_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        labels.data = labels.data - n_net * 10

        labels_hot = torch.eye(NUM_CLASSES)[labels]
        labels_hot = labels_hot.to(DEVICE)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels_hot)

        # statistics
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate average losses
    epoch_loss = running_loss / float(len(test_dataloader.dataset))
    # Calculate Accuracy
    accuracy = running_corrects / float(len(test_dataloader.dataset))

    return accuracy, epoch_loss


# train function
def final_test(net, test_dataloader):
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    for index, images, labels in test_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        # lab = labels.to(DEVICE)

        # labels_hot = torch.eye(NUM_CLASSES * CLASSES_BATCH)[labels]
        # labels_hot = labels_hot.to(DEVICE)

        outputs = []
        loss = []
        idxs = []

        # TODO: change the 10
        # labels_hot = torch.eye(10)[labels]
        # labels_hot = labels_hot.to(DEVICE)
        for i, n in enumerate(net):
            n.to(DEVICE)
            n.train(False)

            # We compute the loss for each output in order to choose the nn
            # with the smallest loss value
            output = n(images)
            outputs.append(output)

            output = torch.softmax(output, 1)
            idxs.append(mIndexFunction(output))

            # loss.append(criterion(output, labels).item())

        best_net_index = np.asarray(idxs).argmax(axis=0)
        # best_net_index = np.asarray(loss).argmin(axis=0)
        preds = classifier(outputs[best_net_index])

        # TODO: overwrite the output with normalized values (for loss function)
        # TODO: Understand what s label and how to adapt to the nn forest
        # TODO: Then write a loss function

        # running_loss += loss[best_net_index] * images.size(0)
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate average losses
    epoch_loss = running_loss / float(len(test_dataloader.dataset))

    # Calculate Accuracy
    accuracy = running_corrects / float(len(test_dataloader.dataset))

    return accuracy, epoch_loss


# TODO: Define a smart classifier
def classifier(outputs):
    _, preds = torch.max(outputs.data, 1)
    return preds


def mIndexFunction(output):
    tot = 0
    idx = []
    output = output.cpu().detach().numpy()
    # print('output', output)
    for out in output:
        out.sort()
        out = out[::-1]
        # print('outreversed', out)

        # tot = 0
        # for i in range(3):#len(out)):
        #     if i == 0: tot = out[i]
        #     tot -= out[i] / float(i + 1)

        tot = 1
        for i in range(1, 3):
            tot -= out[i] ** 2 / (out[0] * i)

        idx.append(tot * out[1] / out[0])
        # print('tot', tot*(out[1])/out[0])
    return np.squeeze(idx)


# train function
def train(net, train_dataloader, test_dataloader, n_net):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    parameters_to_optimize = net.parameters()
    optimizer = optim.SGD(parameters_to_optimize, lr=LR, weight_decay=WEIGHT_DECAY)

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    best_net = []
    best_accuracy = 0

    net.to(DEVICE)

    for epoch in range(NUM_EPOCHS):

        if epoch in STEPDOWN_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / STEPDOWN_FACTOR

        running_loss = 0.0
        running_corrects_train = 0

        for index, inputs, labels in train_dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            labels.data = labels.data - n_net * 10

            labels_hot = torch.eye(NUM_CLASSES)[labels]
            labels_hot = labels_hot.to(DEVICE)

            net.train(True)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels_hot)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data).data.item()

        # Calculate average losses
        epoch_loss = running_loss / float(len(train_dataloader.dataset))
        # Calculate accuracy
        epoch_acc = running_corrects_train / float(len(train_dataloader.dataset))

        if epoch % 10 == 0 or epoch == (NUM_EPOCHS - 1):
            print('Epoch {} Loss:{:.4f} Accuracy:{:.4f}'.format(epoch, epoch_loss, epoch_acc))
            for param_group in optimizer.param_groups:
                print('Learning rate:{}'.format(param_group['lr']))
            print('-' * 30)

        epoch_test_accuracy, epoch_test_loss = test(net, test_dataloader, n_net)

        train_accuracies.append(epoch_acc)
        train_losses.append(epoch_loss)
        test_accuracies.append(epoch_test_accuracy)
        test_losses.append(epoch_test_loss)

        if epoch_test_accuracy > best_accuracy:
            best_accuracy = epoch_test_accuracy
            best_net = copy.deepcopy(net)

    return best_net, train_accuracies, train_losses, test_accuracies, test_losses


def incremental_learning():
    # Define transforms for training phase
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    net = []

    new_acc_train_list = []
    new_loss_train_list = []
    new_acc_test_list = []
    new_loss_test_list = []
    all_acc_list = []

    for i in range(CLASSES_BATCH):
        net.append(resnet32(num_classes=NUM_CLASSES))

        print('-' * 30)
        print(f'**** ITERATION {i + 1} ****')
        print('-' * 30)

        print('Loading the Datasets ...')
        print('-' * 30)

        train_dataset = Cifar100(classes=range(i * 10, (i + 1) * 10), train=True, transform=transform_train)
        test_dataset = Cifar100(classes=range(i * 10, (i + 1) * 10), train=False, transform=transform_test)

        print('-' * 30)
        print('Training ...')
        print('-' * 30)

        # Prepare Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

        net[i], train_accuracies, train_losses, test_accuracies, test_losses = train(net[i], train_dataloader,
                                                                                     test_dataloader, i)

        new_acc_train_list.append(train_accuracies)
        new_loss_train_list.append(train_losses)
        new_acc_test_list.append(test_accuracies)
        new_loss_test_list.append(test_losses)

        print('Testing ...')
        print('-' * 30)

        all_classes_dataset = Cifar100(classes=range(0, (i + 1) * 10), train=False, transform=transform_test)

        # Prepare Dataloader
        test_all_dataloader = DataLoader(all_classes_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                         drop_last=False, num_workers=4)

        print('All classes')

        all_acc_list.append(final_test(net, test_all_dataloader))

        print('-' * 30)

    return new_acc_train_list, new_loss_train_list, new_acc_test_list, new_loss_test_list, all_acc_list

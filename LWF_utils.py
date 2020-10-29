import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from ..Cifar100.utils import Cifar100
from ..resnet_cifar import resnet32

DEVICE = 'cuda'
NUM_CLASSES = 10
BATCH_SIZE = 128
CLASSES_BATCH = 10
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
LR = 2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001
NUM_EPOCHS = 70
LAMBDA = 0.3

def test(net, test_dataloader, n_classes):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    net.to(DEVICE)
    net.train(False)

    running_loss = 0.0
    running_corrects = 0
    for index, images, labels in test_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        labels_hot = torch.eye(n_classes)[labels]
        labels_hot = labels_hot.to(DEVICE)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # loss = criterion(outputs, labels)
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
def update_classes(net, n_new_classes):
    in_features = net.fc.in_features
    out_features = net.fc.out_features
    weight = net.fc.weight.data
    #bias = net.fc.bias.data

    new_out_features = out_features + n_new_classes

    net.fc = nn.Linear(in_features, new_out_features, bias=False)

    net.fc.weight.data[:out_features] = weight
    #net.fc.bias.data[:out_features] = bias

    return net, new_out_features


def train(net, train_dataloader, test_dataloader, n_classes):
    prev_net = copy.deepcopy(net).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    parameters_to_optimize = net.parameters()

    # START
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

            # labels_hot = torch.eye(net.fc.out_features)[labels]
            # labels_hot = labels_hot.to(DEVICE)

            net.train(True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            if n_classes != 10:
                old_outputs = prev_net(inputs)
                # with torch.no_grad():
                #     old_outputs = torch.sigmoid(prev_net(inputs)).to(DEVICE)
                # labels_hot = torch.cat((old_outputs[:, :-10], labels_hot[:, -10:]), 1)
                new_outputs = outputs[:, :-10]
                old_outputs = old_outputs[:, :-10]
                old_loss = MultinomialLogisticLoss(old_outputs, new_outputs)
                loss = LAMBDA * old_loss + loss

            # loss = criterion(outputs, labels_hot)

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

        epoch_test_accuracy, epoch_test_loss = test(net, test_dataloader, n_classes)

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

    net = resnet32(num_classes=NUM_CLASSES)
    n_classes = NUM_CLASSES
    # n_features = net.fc.in_features
    #
    # net = nn.Sequential(*list(net.children())[:-1])
    # net = nn.DataParallel(net)
    #
    # fc = nn.Linear(n_features, 0, bias=False)

    new_acc_train_list = []
    new_loss_train_list = []
    new_acc_test_list = []
    new_loss_test_list = []
    all_acc_list = []

    for i in range(CLASSES_BATCH):
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

        net, train_accuracies, train_losses, test_accuracies, test_losses = train(net, train_dataloader, test_dataloader, n_classes)

        new_acc_train_list.append(train_accuracies)
        new_loss_train_list.append(train_losses)
        new_acc_test_list.append(test_accuracies)
        new_loss_test_list.append(test_losses)

        print('Testing ...')
        print('-' * 30)

        # Creating dataset for test on previous classes
        # previous_classes = np.array([])

        all_classes_dataset = Cifar100(classes=range(0, (i + 1) * 10), train=False, transform=transform_test)

        # Prepare Dataloader
        test_all_dataloader = DataLoader(all_classes_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

        print('All classes')

        all_acc_list.append(test(net, test_all_dataloader, n_classes))

        net, n_classes = update_classes(net, 10)
        print('n_classes', n_classes)

        print('-' * 30)

    return new_acc_train_list, new_loss_train_list, new_acc_test_list, new_loss_test_list, all_acc_list

def MultinomialLogisticLoss(old_outputs, new_outputs):#, T):
    # L = -1/N * sum(N) sum(C) softmax(new_outputs) * log(softmax(old_outputs))

    old_outputs = torch.log_softmax(old_outputs, dim=1)
    new_outputs = torch.softmax(new_outputs, dim=1)
    mean = -torch.mean(torch.sum(old_outputs * new_outputs, dim=1), dim=0)
    #TODO: maybe return Variable(mean.data, requires_grad=True).to(DEVICE)

    return mean

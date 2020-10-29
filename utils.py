from torchvision.datasets import CIFAR100
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

class Cifar100(CIFAR100):
    def __init__(self, root = 'Dataset', classes=range(10), train=True, transform=None, target_transform=None, download=True):
        
        super(Cifar100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # Select subset of classes
        
        data = []
        targets = []

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                data.append(self.data[i])
                targets.append(self.targets[i])

        self.data = np.array(data)
        self.targets = targets


    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
       
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        
        return len(self.data)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def append(self, images, labels):

        self.data = np.concatenate((self.data, images), axis=0)
        self.targets = self.targets + labels


def plot(new_acc_train, new_acc_test, new_loss_train, new_loss_test, all_acc, args):
    num_epochs = len(new_acc_train[0])
    x = np.linspace(1, num_epochs, num_epochs)

    for i, (acc_train, acc_test, loss_train, loss_test) in enumerate(zip(new_acc_train, new_acc_test, new_loss_train, new_loss_test)):

        title = 'Accuracy dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d' \
                % (i + 1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'])
        title2 = 'Loss dataset # %d - BATCH_SIZE= %d LR= %f  EPOCHS= %d' \
                 % (i + 1, args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'])

        plt.plot(x, acc_train, color='mediumseagreen')
        plt.plot(x, acc_test, color='lightseagreen')
        plt.title(title)
        plt.xticks(np.arange(1, num_epochs, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train accuracy', 'Test accuracy'], loc='best')
        plt.show()

        plt.plot(x, loss_train, color='mediumseagreen')
        plt.plot(x, loss_test, color='lightseagreen')
        plt.title(title2)
        plt.xticks(np.arange(1, num_epochs, 4))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train loss', 'Test loss'], loc='best')
        plt.show()

    plt.plot(all_acc, color='lightseagreen')
    plt.title('%s incremental learning accuracy' % (args['name']))
    plt.xticks(np.arange(1, len(all_acc), 1))
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend(['Test accuracy'], loc='best')
    plt.show()

    # csv_name = '%s - BATCH_SIZE= %d LR= %f  EPOCHS= %d' % (args['name'], args['BATCH_SIZE'], args['LR'], args['NUM_EPOCHS'])
    # pd.DataFrame(all_acc).to_csv('./Results/%s.csv' % csv_name)

    print('Accuracy last test', new_acc_test[-1])

def plot2(new_acc_test, all_acc, args):
  
  x = np.linspace(1, 10, 10)

  plt.plot(x, new_acc_test, color='mediumseagreen')
  plt.title('Individual accuracy for each batch of classes')
  plt.xlabel('Batch index')
  plt.ylabel('Accuracy')
  plt.legend(['Test accuracy'], loc='best')
  plt.show()

  x = np.linspace(10, 100, 10)

  plt.plot(x, all_acc, color='lightseagreen')
  plt.title('Incremental learning accuracy')
  plt.xlabel('Number of classes')
  plt.ylabel('Accuracy')
  plt.legend(['Test accuracy'], loc='best')
  plt.show()

  csv_name = 'iCarl_random_NME'
  pd.DataFrame(all_acc).to_csv('%s.csv' % csv_name)          

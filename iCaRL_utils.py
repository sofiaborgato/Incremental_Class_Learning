import numpy as np
import time
import random 
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

from torchvision import transforms

from ..Cifar100.utils import Cifar100
from ..resnet_cifar import resnet32

#bisogna importare resnet e le utils con dataset e funzioni di plot######################################################

class iCaRL():
    
    def __init__(self, memory=2000, params=None):
        self.memory = memory
        self.params = params
        self.device = 'cuda'
        
    def different_classifier(self, data, exemplars, net, classifier):
      
      print('-'*30)
      print(f'**** Classification: Different classifier ****')
      print('-'*30)
      
      train_data = []
      targets = []

      print(f'**** fitting the classifier on the exemplars... ****')
      print('-'*30)
      for key in exemplars:

        loader = DataLoader(exemplars[key], batch_size=1024, shuffle=False, num_workers=4, drop_last=False)
        mean = torch.zeros((1,64),device=self.device)

        for _, images, labels in loader:
          with torch.no_grad():

            images = images.to(self.device)
            outputs = net(images,features=True)
            
            for output,label in zip(outputs,labels):
              train_data.append(np.array(output.cpu()))
              targets.append(np.array(label))
    
      classifier.fit(train_data, targets)

      loader = DataLoader(data, batch_size=1024, shuffle=False, num_workers=4, drop_last=False)

      running_correct = 0.0
      print(f'**** predicting... ****')
      print('-'*30)

      for _, images, labels in loader:
        
        images = images.to(self.device)
        
        with torch.no_grad():

          outputs = net(images,features=True)
          preds = []

          for output in outputs:

            pred = classifier.predict([np.array(output.cpu())])
            preds.append(pred)
          
          for label, pred in zip(labels, preds):
            if label == pred[0]:
              running_correct += 1
      
      accuracy = running_correct/len(data)
      print('Accuracy:{:.4f}'.format(accuracy))

      return accuracy    

    def NME(self, data, exemplars, net, n_classes):
      print('-'*30)
      print(f'**** Classification: NME ****')
      print('-'*30)
      
      means = dict.fromkeys(np.arange(n_classes))
      net.eval()

      # compute exemplars prototypes
      print(f'**** computing means of exemplars... ****')
      print('-'*30)
      for key in exemplars:
        
        loader = DataLoader(exemplars[key], batch_size=1024, shuffle=False, num_workers=4, drop_last=False)
        mean = torch.zeros((1,64), device=self.device)
        
        for _, images, _ in loader:
          with torch.no_grad():

            images = images.to(self.device)
            outputs = net(images, features=True)
            
            for output in outputs:
              mean += output
        
        mean = mean / len(exemplars[key])
        means[key] = mean / mean.norm()

      # applying nme classification
      print(f'**** predicting... ****')
      print('-'*30)

      loader = DataLoader(data, batch_size=1024, shuffle=False, num_workers=4, drop_last=False)

      running_correct = 0.0
      for _, images, labels in loader:

        images = images.to(self.device)
        
        with torch.no_grad():
          
          outputs = net(images, features=True)
          preds = []
          
          for output in outputs:
            
            pred = None
            min_dist = float('inf')
            
            for key in means:
              dist = torch.dist(means[key], output)
              if dist < min_dist:
                min_dist = dist
                pred = key
            
            preds.append(pred)
          
          for label, pred in zip(labels,preds):
            if label == pred:
              running_correct += 1
      
      accuracy = running_correct / len(data)
      print('Accuracy:{:.4f}'.format(accuracy))

      return accuracy

    #train function for update_representation returns the new net and its training losses for epoch
    def train(self, net, old_net, train_dataloader, optimizer, n_epochs, n_classes):

      criterion = nn.BCEWithLogitsLoss()
      parameters_to_optimize = net.parameters()

      train_losses = []

      net.to(self.device)

      for epoch in range(n_epochs):

        if epoch in self.params['STEPDOWN_EPOCHS']:
          for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/self.params['STEPDOWN_FACTOR']

        running_loss = 0.0

        for indexes, inputs, labels in train_dataloader:
          inputs = inputs.to(self.device)
          labels = labels.to(self.device)
          
          labels_hot=torch.eye(n_classes)[labels]
          labels_hot = labels_hot.to(self.device)

          net.train(True)
          # zero the parameter gradients
          optimizer.zero_grad()
          # forward
          outputs = net(inputs)

          if n_classes == 10:
            loss = criterion(outputs[:, n_classes - 10:], labels_hot[:, n_classes - 10:])
          else:
            old_outputs = self.get_old_outputs(inputs, old_net)
            targets = torch.cat((old_outputs, labels_hot[:, n_classes - 10:]), 1)
            loss = criterion(outputs, targets)

          loss.backward()
          optimizer.step()

          # statistics
          running_loss += loss.item() * inputs.size(0)

        # Calculate average losses
        epoch_loss = running_loss / float(len(train_dataloader.dataset))
        
        if epoch % 10 == 0 or epoch == (n_epochs-1):
          print('Epoch {} Loss:{:.4f}'.format(epoch, epoch_loss))
          for param_group in optimizer.param_groups:
            print('Learning rate:{}'.format(param_group['lr']))
          print('-'*30)

        train_losses.append(epoch_loss)

      return net, train_losses

 
    def update_representation(self, new_data, exemplars, net, n_classes):
        
        print('-'*30)
        print(f'**** Update Representation... ****')
        print('-'*30)
        
        # concatenate new data with set of exemplars
        if len(exemplars) != 0:
          data = new_data + exemplars
        else:
          data = new_data
        
        old_net = deepcopy(net)
        
        loader = DataLoader(data, batch_size=self.params['BATCH_SIZE'], shuffle=True, num_workers=4, drop_last=True)
        if n_classes != 10:
          # update net last layer
          net = self.update_net(net, n_classes)
          
        optimizer = torch.optim.SGD(net.parameters(), lr=self.params['LR'], momentum=self.params['MOMENTUM'], weight_decay=self.params['WEIGHT_DECAY'])
        
        net, train_losses = self.train(net, old_net, loader, optimizer, self.params['NUM_EPOCHS'], n_classes)

        return net, train_losses

    def random_exemplar(self, data, n_classes):
      print('-'*30)
      print(f'**** Construct new exemplars: Random mode ****')
      print('-'*30)

      m = int(self.memory / n_classes)

      sample_per_class = dict.fromkeys(np.arange(n_classes - 10, n_classes))
      exemplars = dict.fromkeys(np.arange(n_classes - 10, n_classes))

      for label in sample_per_class:
          sample_per_class[label] = []
          exemplars[label] = []

      for item in data:
          for label in sample_per_class:
            if item[2] == label:
              sample_per_class[label].append(item)

      for label in range(n_classes - 10, n_classes):
        
        indexes = random.sample(range(len(sample_per_class[label])), m)
        
        for i in indexes:
          exemplars[label].append(sample_per_class[label][i])

      return exemplars
    
    def herding_exemplar(self, data, n_classes, net):
        print('-'*30)
        print(f'**** Construct new exemplars: Herding mode ****')
        print('-'*30)

        m = int(self.memory / n_classes)

        means = dict.fromkeys(np.arange(n_classes - 10, n_classes))
        sample_per_class = dict.fromkeys(np.arange(n_classes - 10, n_classes))
        exemplars = dict.fromkeys(np.arange(n_classes - 10, n_classes))

        for label in sample_per_class:
          sample_per_class[label] = []
          exemplars[label] = []
          means[label] = []
        
        for item in data:
          for label in sample_per_class:
            if item[2] == label:
              sample_per_class[label].append(item)
        
        # generate new exemplars
        net.eval()
        for label in sample_per_class:
          # initialize mean tensor, is a single value with a number of components equal to the number of outputs of the last conv layer of the resnet32
          mean = torch.zeros((1,64), device=self.device)
          data_features = []
          
          # compute means of data features for each class 
          with torch.no_grad():
            loader = DataLoader(sample_per_class[label], batch_size=1024, shuffle=False, num_workers=4, drop_last=False)
            for _, images, _ in loader:
                
                images = images.to(self.device)
                outputs = net(images,features=True)
                
                for output in outputs:
                    output = output.to(self.device)
                    
                    # save the data features to use them also in the examplar selection
                    data_features.append(output)
                    mean += output
            
            mean = mean / len(sample_per_class[label])
            # normalize the mean
            means[label] = mean / mean.norm()
          
          # find the m sample features which mean is the closest to the one of the entire class
          
          exemplars_features = []
          min_index = 0
          for i in range(m):
            
            min_distance = float('inf')
            exemplar_sum = sum(exemplars_features)
            
            for idx, data_feature in enumerate(data_features):
              
              tmp_mean = (exemplar_sum + data_feature) / (len(exemplars_features) + 1)
              # normalize the mean
              tmp_mean = tmp_mean / tmp_mean.norm()
              
              if torch.dist(mean, tmp_mean) < min_distance:
                min_distance = torch.dist(mean, tmp_mean)
                min_index = idx
               
            exemplars[label].append(sample_per_class[label][min_index])
            exemplars_features.append(data_features[min_index])
            sample_per_class[label].pop(min_index)
            data_features.pop(min_index)

        return exemplars

    def reduce_exemplar(self, exemplars, n_classes):
      print('-'*30)
      print(f'**** Reduce old classes exemplar sets... ****')
      print('-'*30)

      m = int(self.memory / n_classes)

      for key in exemplars:
        exemplars[key] = exemplars[key][:m]
      
      return exemplars
    
    def get_old_outputs(self, images, net):
      # Forward pass in the old network
      
      net.eval()

      with torch.no_grad():
        images = images.to(self.device)  
        out = torch.sigmoid(net(images))

      out = out.to(self.device)

      return out

    def update_net(self, net, n_classes):
      in_features = net.fc.in_features
      out_features = net.fc.out_features
      weight = net.fc.weight.data
      bias = net.fc.bias.data

      net.fc = nn.Linear(in_features, n_classes)
      net.fc.weight.data[:out_features] = weight
      net.fc.bias.data[:out_features] = bias

      return net
    
    # run iCarl routine
    def run(self, herding=True):
      
      exemplars = {}
      new_exemplars = []
      exemplars_as_list = []
      accuracy_new = []
      accuracy_all = []

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

      net = resnet32()

      for i in range(int(self.params['NUM_CLASSES']/self.params['CLASSES_BATCH'])):
        print('-'*30)
        print(f'**** ITERATION {i+1} ****')
        print('-'*30)

        n_classes = (i+1)*10

        train_dataset = Cifar100(classes=range(i*10, (i + 1)*10), train=True, transform=transform_train)
        test_dataset = Cifar100(classes=range(i*10, (i + 1)*10), train=False, transform=transform_test)

        # update representation
        net, train_losses = self.update_representation(train_dataset, exemplars_as_list, net, n_classes)

        # here we can plot the train losses if we want

        # update exemplar sets
        exemplars = self.reduce_exemplar(exemplars, n_classes)
        
        if herding:
          new_exemplars = self.herding_exemplar(train_dataset, n_classes, net)
        else:
          new_exemplars = self.random_exemplar(train_dataset, n_classes)
        
        exemplars.update(new_exemplars)

        exemplars_as_list = [item for class_exemplars in exemplars.values() for item in class_exemplars]

        if(classifier is None):
          # compute accuracy on the new class batch
          accuracy_new.append(self.NME(test_dataset, exemplars, net, n_classes))

          # compute accuracy on all the classes seen so far
          test_dataset_sofar = Cifar100(classes=range(0, (i + 1)*10), train=False, transform=transform_test)
          accuracy_all.append(self.NME(test_dataset_sofar, exemplars, net, n_classes))
        else:
          accuracy_new.append(self.different_classifier(test_dataset, exemplars, net, classifier))

          # compute accuracy on all the classes seen so far
          test_dataset_sofar = Cifar100(classes=range(0, (i + 1)*10), train=False, transform=transform_test)
          accuracy_all.append(self.different_classifier(test_dataset_sofar, exemplars, net, classifier))

      return accuracy_new, accuracy_all

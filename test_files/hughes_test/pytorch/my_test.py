
import torch
import torch.utils.data.dataset as Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

PATH_TO_INPUT = '/media/tdh5188/easystore/data/test_pytorch_data'

print('preprocessing')


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_set = dsets.CIFAR10(root=PATH_TO_INPUT,
                          train=True,
                          download=True,
                          transform=transform)


trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

valid_set = dsets.CIFAR10(root=PATH_TO_INPUT,
                          train=False,
                          download=True,
                          transform=transform)


validloader = torch.utils.data.DataLoader(valid_set,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



######################################################################

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))


#get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


#imshow(torchvision.utils.make_grid(images))

#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


######################################################################



## Defining a Convolution Nerual Network


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

print('creating model')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,64)
        self.fc3 = nn.Linear(64,10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
model = Model()


######################################################################


##define a loss function and optimizer

import torch.optim as optim

print('defining loss function')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


######################################################################

##train the network

print('training network')

def train (n_epochs):
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            #get the input
            inputs, labels = data
            
            #wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            
            #zero the parameter gradients
            optimizer.zero_grad()
            
            #forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print statistics
            running_loss += loss.data[0]
    #        if i % 2000 == 1999:
    #           print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 2000)) 
    #           
    #           running_loss = 0.0
    
        print('progress:',epoch, ':: loss =', loss.data[0])

#train(2)

print('Finished Training')



######################################################################

##test the network on the test data

#dataiter = iter(validloader)
#images, labels = dataiter.next()

#print images
#imshow(torchvision.utils.make_grid(images))

#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))



































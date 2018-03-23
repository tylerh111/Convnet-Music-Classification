import numpy as np
import pandas as pd
import os
import sys
import random
from PIL import Image
from skimage.transform import resize
from random import shuffle,seed,randint
import time
import shutil
import argparse
import torch



parser = argparse.ArgumentParser(description='PyTorch ImageNet (modified) Training')

parser.add_argument('-d', '--data', default='/media/tdh5188/easystore/data/convnet_input/input/',
                    type=str, metavar='V', help='path to dataset')

parser.add_argument('-v', '--version-num', default='0', type=str,
                    metavar='V', help='version number for training')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')

parser.add_argument('-l', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

args = parser.parse_args()


######################################################################

# hyper param and param getting from command line arguments


VERSION = args.version_num

SCRIPT_NAME = "spectrogram_v_" + VERSION
if os.path.exists(SCRIPT_NAME + "/"):
    print("Directory for saved models already exists:",SCRIPT_NAME)
    sys.exit(-1)
else:
    os.makedirs(SCRIPT_NAME + "/")

STARTING_MODEL = args.resume


PATH = args.data
SEEDS = [0,0,0]

#print(SCRIPT_NAME)
#print("-- seeds:", SEEDS)

# Set random seed so that the results are repeatable
seed(SEEDS[0])
np.random.seed(SEEDS[1])



# other preset params
start_epoch = args.start_epoch
print_freq = args.print_freq
best_prec1 = 0


# hyper parameters, parameters, and knobs
print('-- setting hyper parameters')

# class list and dictionary
list_classes = [
    'Rock',
    'Classical'
]

dict_classes = {
    'Rock':0,
    'Classical':1
}

n_class = len(list_classes)

# hyper params
n_epochs = args.epochs

batch_size = args.batch_size
#batch_per_epoch = 80

learning_rate = args.learning_rate
momentum = args.momentum
weight_decay = args.weight_decay
dampening = 0
#nesterov = False


#params
dim_x = 224
dim_y = 224
dim_z = 3
margin = 100
#random_location = True



def print_info():
    print('================================')
    print(SCRIPT_NAME)
    print('-------------------')
    print('Seeds:\t\t', SEEDS)
    print('Data Path:\t', PATH)
    print('Start Model:\t', STARTING_MODEL)
    print('Epochs:\t\t [' + str(start_epoch) + ' : ' + str(start_epoch + n_epochs) + ']')
    print('Batch Size:\t', batch_size)
    print('Learn Rate:\t', learning_rate)
    print('Momentum:\t', momentum)
    print('Dampening:\t', dampening)
    print('Dimensions:\t', (dim_x, dim_y, dim_z))
    print('Margin:\t\t', margin)
    print('================================')

print_info()

#######################################################################

## model creation

print('-- creating model')

import torch.nn as nn
import torch.nn.functional as functional


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1,16 * 53 * 53)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Model()
#import torchvision.models.resnet as resnet
#model = resnet

######################################################################

# helpful functions

# def get_class_from_path(filepath):
# 	return os.path.dirname(filepath).split(os.sep)[-1]
#
#
# def label_transform(label):
# 	return dict_classes[label]


# show PIL image
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))


######################################################################

# loss and optimizer
print('-- initializing criterion and loss function')

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(),
                lr = learning_rate,
                momentum = momentum,
                weight_decay = weight_decay,
                dampening = dampening)


# optionally resume from a checkpoint

if STARTING_MODEL is not "":
    if os.path.isfile(STARTING_MODEL):
        print("==> loading checkpoint '{}'".format(STARTING_MODEL))
        checkpoint = torch.load(STARTING_MODEL)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("==> loaded checkpoint '{}' (epoch {})".format(STARTING_MODEL,checkpoint['epoch']))
    else:
        print("==> no checkpoint found at '{}'".format(STARTING_MODEL))

######################################################################

## dataset creation
print('-- creating datasets')

from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# #load_images reads in a picture and crops the image to specification
# def load_images(filepath):
# 	return Image.open(filepath)

print('==> creating training dataset')
train_dataset = dsets.ImageFolder(root = PATH + 'train/',
                                  transform = transforms.Compose([
                                      transforms.RandomCrop((dim_x,dim_y),margin),
                                      #transforms.RandomResizedCrop((dim_x,dim_y)),
                                      #transforms.RandomHorizontalFlip(),

                                      transforms.ToTensor(),
                                      #transforms.Normalize((.,.,.),(.,.,.))
                                  ]))

print('==> creating validation dataset')
valid_dataset = dsets.ImageFolder(root = PATH + 'valid/',
                                  transform = transforms.Compose([
                                      transforms.RandomCrop((dim_x,dim_y),margin),
                                      #transforms.RandomResizedCrop((dim_x,dim_y)),
                                      #transforms.RandomHorizontalFlip(),

                                      transforms.ToTensor(),
                                      #transforms.Normalize((.,.,.),(.,.,.))
                                  ]))

# creating data loaders
print('-- creating data loaders')

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = batch_size,
                          shuffle = True)

valid_loader = DataLoader(dataset = valid_dataset,
                          batch_size = batch_size,
                          shuffle = False)


######################################################################

# procedures for training and validating

# adapted from ImageNet
# https://github.com/pytorch/examples/blob/master/imagenet/

from torch.autograd import Variable

def train(train_loader,model,criterion,optimizer,epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i,(input,target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = target.cuda(async = True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        #print('\t==>>input_var =',input_var)
        output = model(input_var)
        loss = criterion(output,target_var)

        # measure accuracy and record loss
        prec1,prec5 = accuracy(output.data,target,topk = (1,5))
        losses.update(loss.data[0],input.size(0))
        top1.update(prec1[0],input.size(0))
        top5.update(prec5[0],input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch,i,len(train_loader),batch_time = batch_time,
                data_time = data_time,loss = losses,top1 = top1,top5 = top5))


def validate(valid_loader,model,criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i,(input,target) in enumerate(valid_loader):
        #target = target.cuda(async = True)
        input_var = torch.autograd.Variable(input,volatile = True)
        target_var = torch.autograd.Variable(target,volatile = True)

        # compute output
        output = model(input_var)
        loss = criterion(output,target_var)

        # measure accuracy and record loss
        prec1,prec5 = accuracy(output.data,target,topk = (1,5))
        losses.update(loss.data[0],input.size(0))
        top1.update(prec1[0],input.size(0))
        top5.update(prec5[0],input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i,len(valid_loader),batch_time = batch_time,loss = losses,
                top1 = top1,top5 = top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1 = top1,top5 = top5))

    return top1.avg


def save_checkpoint(state, is_best, filepath = SCRIPT_NAME, filename = 'checkpoint.pth.tar'):
    torch.save(state,filepath + filename)
    if is_best:
        shutil.copyfile(filename, filepath + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer,epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output,target,topk = (1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim = True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


######################################################################

# training
print('-- STARTING TRAINING')

for epoch in range(start_epoch,start_epoch + n_epochs):
    adjust_learning_rate(optimizer,epoch)

    # train for one epoch
    train(train_loader,model,criterion,optimizer,epoch)

    # evaluate on validation set
    prec1 = validate(valid_loader,model,criterion)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1,best_prec1)
    save_checkpoint({
        'epoch':epoch + 1,
        #'arch':args.arch,
        'state_dict':model.state_dict(),
        'best_prec1':best_prec1,
        'optimizer':optimizer.state_dict(),},
        is_best,
        SCRIPT_NAME,
        'checkpoint_e' + str(epoch) + '.pth.tar')







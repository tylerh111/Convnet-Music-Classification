
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
import torch

if not (1 <= len(sys.argv) <= 2):
	print("Usage: python",sys.argv[0],"VERSION") # [STARTING_MODEL]")
	sys.exit(-1)


VERSION = sys.argv[1]

SCRIPT_NAME = "spectrogram_v" + VERSION
if os.path.exists(SCRIPT_NAME + "/"):
	print("Directory for saved models already exists:",SCRIPT_NAME)
	sys.exit(-1)

STARTING_MODEL = ""
if len(sys.argv) == 2:
	STARTING_MODEL = sys.argv[2]

PATH = "/media/tdh5188/easystore/data/convnet_input/input/" # path to training files
#DIR_SEP = "/" # "/" for unix, "\\" for windows
#SEEDS = [randint(0,10000),randint(0,10000),randint(0,10000)]
SEEDS = [0,0,0]

print(SCRIPT_NAME)
#print("Model:",MODEL)
#print("Validation split:", VALID_SPLIT)
#print("Starting Model:",STARTING_MODEL)
print("Seeds:",SEEDS)

# Set random seed so that the results are repeatable
seed(SEEDS[0])
np.random.seed(SEEDS[1])
#set_random_seed(SEEDS[2])



# other preset params
start_epoch = 0
print_freq = 100



#######################################################################

# hyper parameters, parameters, and knobs

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
n_epochs = 3

batch_size = 16
#batch_per_epoch = 80

learning_rate = 0.001
momentum = 0.9
weight_decay = 0
dampening = 0
#nesterov = False


#params
dim_x = 224
dim_y = 224
dim_z = 3
margin = 100
#random_location = True



#######################################################################

## model creation

import torch.nn as nn
import torch.nn.functional as F

print('creating model')


class Model(nn.Module):
	def __init__(self):
		super(Model,self).__init__()
		self.conv1 = nn.Conv2d(3,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16 * 5 * 5,120)
		self.fc2 = nn.Linear(120,64)
		self.fc3 = nn.Linear(64,10)

	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


model = Model()

######################################################################

# helpful functions

def get_class_from_path(filepath):
	return os.path.dirname(filepath).split(os.sep)[-1]


def label_transform(label):
	return dict_classes[label]


# show PIL image
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	img = img / 2 + 0.5  #unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))


######################################################################

# loss and optimizer
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
		print("=> loading checkpoint '{}'".format(STARTING_MODEL))
		checkpoint = torch.load(STARTING_MODEL)
		start_epoch = checkpoint['epoch']
		#best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format(STARTING_MODEL, checkpoint['epoch']))
	else:
		print("=> no checkpoint found at '{}'".format(STARTING_MODEL))


######################################################################

## dataset creation

from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# #load_images reads in a picture and crops the image to specification
# def load_images(filepath):
# 	return Image.open(filepath)


train_dataset = dsets.ImageFolder(root = PATH + 'train/',
								  transform = transforms.Compose([
									  transforms.RandomCrop((dim_x,dim_y),margin),
									  #transforms.RandomResizedCrop((dim_x,dim_y)),
									  #transforms.RandomHorizontalFlip(),

									  transforms.ToTensor(),
									  #transforms.Normalize((.,.,.),(.,.,.))
								  ]))

valid_dataset = dsets.ImageFolder(root = PATH + 'valid/',
								  transform = transforms.Compose([
									  transforms.RandomCrop((dim_x,dim_y),margin),
									  #transforms.RandomResizedCrop((dim_x,dim_y)),
									  #transforms.RandomHorizontalFlip(),

									  transforms.ToTensor(),
									  #transforms.Normalize((.,.,.),(.,.,.))
								  ]))

# creating data loaders

train_loader = DataLoader(dataset = train_dataset,
						  batch_size = batch_size,
						  shuffle = True)

valid_loader = DataLoader(dataset = valid_dataset,
						  batch_size = batch_size,
						  shuffle = False)


######################################################################



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


def save_checkpoint(state,is_best,filename = 'checkpoint.pth.tar'):
	torch.save(state,filename)
	if is_best:
		shutil.copyfile(filename,'model_best.pth.tar')


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
from torch.autograd import Variable


for epoch in range(start_epoch, start_epoch + n_epochs):
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
		'optimizer':optimizer.state_dict(),
	},is_best)




















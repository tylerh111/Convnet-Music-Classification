
import numpy as np
import pandas as pd
import os
import sys
import random
from PIL import Image
from skimage.transform import resize
from random import shuffle,seed,randint

if not (1 <= len(sys.argv) <= 1):
	print("Usage: python",sys.argv[0],"VERSION") # [STARTING_MODEL]")
	sys.exit(-1)


VERSION = sys.argv[1]

SCRIPT_NAME = "spectrogram_v" + VERSION
if os.path.exists(SCRIPT_NAME + "/"):
	print("Directory for saved models already exists:",SCRIPT_NAME)
	sys.exit(-1)

#STARTING_MODEL = ""
#if len(sys.argv) == 4:
#	STARTING_MODEL = sys.argv[3]

PATH = "/media/tdh5188/easystore/data/convnet_input" # path to training files
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



#######################################################################

## dataset creation


#load_images reads in a picture and crops the image to specification
def load_images(filepath): #, left = None,top = None,random = True,margin = 0,width = 112,height = 112):
	#im_array = np.array(Image.open(filepath),dtype = "uint8")
	#pil_im = Image.fromarray(im_array)

	img = Image.open(filepath)
	#arr = np.array(img, dtype = "uint8")

	if left is None:
		if random:
			left = randint(margin,img.size[0] - margin - width + 1)
		else:
			left = (img.size[0] - width) // 2
	if top is None:
		if random:
			top = randint(margin,img.size[1] - margin - height + 1)
		else:
			top = (img.size[1] - height) // 2
	new_array = np.array(img.crop((left,top,left + width,top + height)))
	return new_array / 255


def transform_image(img, left=None, top=None, random=True, margin=0, width=112, height=112):
	if left is None:
		if random:
			left = randint(margin,img.size[0] - margin - width + 1)
		else:
			left = (img.size[0] - width) // 2
	if top is None:
		if random:
			top = randint(margin,img.size[1] - margin - height + 1)
		else:
			top = (img.size[1] - height) // 2
	new_array = np.array(img.crop((left,top,left + width,top + height)))
	return new_array / 255



def load_sets(n_class):
	list_paths = []
	for subdir,dirs,files in os.walk(PATH + "/input"):
		for file in files:
			filepath = subdir + "/" + file
			list_paths.append(filepath)

	# Grab file list and labels

	train_set = [[] for i in range(n_class)]
	valid_set = [[] for i in range(n_class)]

	for filepath in list_paths:
		label = label_transform(get_class_from_path(filepath))
		# randomly decide to add to training set or validation set
		if random.uniform(0,1) >= 0.10:
			train_set[label].append((filepath, label))
		else:
			valid_set[label].append((filepath, label))

	return {'train':train_set,
			'valid':valid_set}


from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class DataGenerator(Dataset):

	def __init__(self, root, loader, img_paths, transform = None, target_transform = None):
		self.root = root
		self.img_paths = img_paths
		self.classes = list_classes
		self.class_to_idx = dict_classes
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader


	def __getitem__(self,index):
		path, target = self.img_paths[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			img = self.target_transform

		return img, target

	def __len__(self):
		return len(self.img_paths)




######################################################################

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
	img = img / 2 + 0.5  #unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))


#######################################################################


## model creation

from torch.autograd import Variable
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

#######################################################################

## Class list and dictionary
list_classes = [
	'Rock',
	'Classical'
]

dict_classes = {
	'Rock':0,
	'Classical':1
}

n_class = len(list_classes)


def get_class_from_path(filepath):
	return os.path.dirname(filepath).split(os.sep)[-1]


def label_transform(label):
	return dict_classes[label]


# hyper params
n_epochs = 3

batch_size = 16
batch_per_epoch = 80

learning_rate = 0.001

partition = load_sets(n_class = n_class)

train_set = DataGenerator(root = PATH,
						  loader = load_images,
						  img_paths = partition['train'])

valid_set = DataGenerator(root = PATH,
						  loader = load_images,
						  img_paths = partition['valid'])

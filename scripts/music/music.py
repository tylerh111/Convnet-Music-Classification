#################################################
## python script.py <version> [previous model] ##
#################################################

# CNN with a custom dataflow
#
# Author: tdr5143
#
# Each epoch the CNN training takes a random patch from training and validation images to minimize overfitting
# History: used Dr. Blum file to create

import soundfile as sf
import numpy as np
import pandas as pd
import os
import random

from random import shuffle, seed, choice, randint
from tensorflow import set_random_seed
from keras.applications import imagenet_utils,densenet,inception_resnet_v2,resnet50,inception_v3,mobilenet
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Reshape
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras import activations
import sys

if not (2 <= len(sys.argv) <= 3):
    print("Usage: python",sys.argv[0],"MODEL VERSION [STARTING_MODEL]")
    sys.exit(-1)

MODELS = {
    'densenet121':{
        'size':224,
        'preprocessor':densenet.preprocess_input,
    },
    'densenet169':{
        'size':224,
        'preprocessor':densenet.preprocess_input,
    },
    'densenet201':{
        'size':224,
        'preprocessor':densenet.preprocess_input,
    },
    'inceptionresnet':{
        'size':299,
        'preprocessor':inception_resnet_v2.preprocess_input,
    },
    'inception':{
        'size':299,
        'preprocessor':inception_v3.preprocess_input,
    },
    'mobilenet':{
        'size':224,
        'preprocessor':mobilenet.preprocess_input,
    },
    'resnet':{
        'size':224,
        'preprocessor':resnet50.preprocess_input,
    },
}



VERSION = sys.argv[1]

SCRIPT_NAME = "music_v" + VERSION
if os.path.exists(SCRIPT_NAME + "/"):
    print("Directory for saved models already exists:",SCRIPT_NAME)
    sys.exit(-1)

STARTING_MODEL = ""
#if len(sys.argv) == 2:
#    STARTING_MODEL = sys.argv[3]

#PATH = "your/path/to/directory_with_input/
#PATH = "/media/tdh5188/easystore/convnet_input" # path to training files
#DIR_SEP = "/" # "/" for unix, "\\" for windows
SEEDS = [randint(0,10000),randint(0,10000),randint(0,10000)]

print(SCRIPT_NAME)
#print("Model:",MODEL)
#print("Validation split:", VALID_SPLIT)
print("Starting Model:",STARTING_MODEL)
print("Seeds:",SEEDS)

# Set random seed so that the results are repeatable
seed(SEEDS[0])
np.random.seed(SEEDS[1])
set_random_seed(SEEDS[2])

# ## Generate Training and Validation splits
#
# So that they are compatible with the custom dataflow generator

# Any results you write to the current directory are saved as output.
list_paths = []
for subdir,dirs,files in os.walk("E:\\MusicNetworks\\input"):
    for file in files:
        filepath = subdir + "\\" + file
        #print("function filepath: " + filepath)
        list_paths.append(filepath)


######  CHANGE THESE TO ROCK & CLASSICAL
list_classes = [
    'Rock',
    'Classical'
]


dict_classes = {
    'Rock':0,
    'Classical':1
}



def get_class_from_path(filepath):
    #print("function filepath: " + filepath + " label " + os.path.dirname(filepath).split(os.sep)[-1])
    return os.path.dirname(filepath).split(os.sep)[-1]


def label_transform(label):
    return dict_classes[label]


##############################################################################################


#Take a potentially random patch of song, modified by tdr5143
def read_and_crop(filepath,random = True,margin = 0,width = 10000):
    #print("filepath: "+filepath)
    song = sf.read(filepath)
    
    rightChannel = song[0].transpose()[0]
    
    startingPoint = randint(0,len(rightChannel)-width)
    
    fragment = rightChannel[startingPoint:startingPoint+width]
    
    for i in range(0,len(fragment)):
        fragment[i] = (fragment[i]+1)/2

    return fragment


##############################################################################################


# Grab file list and labels

train_set = [[],[]]
valid_set = [[],[]]

for filepath in list_paths:
        label = label_transform(get_class_from_path(filepath))
        # randomly decide to add to training set or validation set
        if random.uniform(0,1) >= 0.10:
                train_set[label].append(filepath)
        else:
                valid_set[label].append(filepath)


partition = {'train': train_set,
             'valid': valid_set}

print("Assembled raining and validation sets complete")



### test prints of file structure
def pretty_print_path():
    print("partition {")
    for mset in partition:
        print("\t",mset,"[")
        for ndx in range(0, len(partition[mset])):
            label = list_classes[ndx]
            print("\t\t",label,"[")
            for filepath in partition[mset][ndx]:
                print("\t\t\t",filepath)
            print("\t\t]")
        print("\t]")
    print("}")


#pretty_print_path()


##########################################################################################################################


## Custom Dataflow Generator, rewritten by tdr5143


class DataGenerator(object):
    def __init__(self,dim_x,batch_size,batches_per_epoch, margin = 100,random_location = True,
                 nclass = 2):
        self.dim_x = dim_x

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.random_location = random_location
        self.margin = margin
        self.nclass = nclass

    def generate(self,list_IDs):
        # Generates batches of samples
        # Infinite loop
        while 1:
            print("Generating batch");
            # Generate batches
            imax = self.batches_per_epoch
            for i in range(imax):
                # Generate data
                X,y = self.__data_generation(list_IDs)

                yield X,y

    def __data_generation(self,list_IDs):
      
        X = np.empty((self.batch_size,self.dim_x))
        y = np.empty((self.batch_size),dtype = int)

        for i in range(self.batch_size):
            
            
                        #choose random label from file_path
            label = randint(0,self.nclass - 1)
            #print("label: " + str(label))
            
            #choose random item from within that label
            sector = randint(0,len(list_IDs[label]) - 1)
            #print("sector: " + str(sector))

       
            #print("path: " + list_IDs[label][sector])
                        #read_and_crop gets image and then copies everything (that the ':' in the array) into a sample index ('i')
            X[i] = read_and_crop(list_IDs[label][sector], #issue here?
                                                   margin = self.margin,
                                                   random = self.random_location,
                                                   width  = self.dim_x)
            y[i] = label

        return X,sparsify(y)

    ## dont need this (used by data generator)


def sparsify(y):
    # Returns labels in binary NumPy array'
    n_classes = 2
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])


#################################################################################################################################################


# ## Train the CNN

from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping,ReduceLROnPlateau,TensorBoard
from keras import optimizers,losses,activations,models
from keras.layers import Convolution2D,Dense,Input,Flatten,Dropout,MaxPooling2D,BatchNormalization,GlobalMaxPool2D,Concatenate
from keras.models import Model

nclass = 2


def get_model():
    num_classes = 2

    input_shape = (10000,) #1D vector instead of picture, perhaps size of 200,000?
    #preprocess = imagenet_utils.preprocess_input

    input_image = Input(shape = input_shape)

    

    x = input_image

    x = Reshape((-1,))(x) #what does this do?
    
    x = Dense(16,activation = 'relu',name = 'fc1')(x)
    #x = Dropout(0.3,name = 'dropout_fc1')(x)
    x = Dense(16,activation = 'relu',name = 'fc2')(x)
    #x = Dropout(0.3,name = 'dropout_fc2')(x)
    prediction = Dense(nclass,activation = "sigmoid",name = "predictions")(x)

    # this is the model we will train
    my_model = Model(inputs = (input_image),outputs = prediction)

    # first: train only the top layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    opt = "rmsprop"
    my_model.compile(optimizer = opt,loss = 'binary_crossentropy',metrics = ['acc'])

    my_model.summary()
    return my_model


########################################################################################################


print("Getting model")
model = get_model()
print("done")

if not os.path.exists(SCRIPT_NAME + "\\"):
    os.makedirs(SCRIPT_NAME + "\\")
file_path = SCRIPT_NAME + "\\weights.{epoch:04d}.hdf5"

callbacks_list = [ModelCheckpoint(file_path,monitor = 'val_acc',verbose = 1)]

# Might need to change this to fit keras data generator
# Parameters
paramsTrain = {
    'dim_x':10000,

    'batch_size':8,
    'batches_per_epoch':150,
    'nclass':2,
    'margin':100,
    'random_location':True
}
paramsValid = {
    'dim_x':10000,

    'batch_size':8,
    'batches_per_epoch':80,
    'nclass':2,
    'margin':100,
    'random_location':True
}

print("starting training")
if STARTING_MODEL != "":
    model.load_weights(STARTING_MODEL)
    
training_generator = DataGenerator(**paramsTrain).generate(partition['train'])
validation_generator = DataGenerator(**paramsValid).generate(partition['valid'])

print("BEGIN")
# Train model on dataset
history = model.fit_generator(generator = training_generator,
                              steps_per_epoch = paramsTrain['batches_per_epoch'],
                              validation_data = validation_generator,
                              validation_steps = paramsValid['batches_per_epoch'],
                              epochs = 10,
                              verbose = 2,callbacks = callbacks_list)

print(history)

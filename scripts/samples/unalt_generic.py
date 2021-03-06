# CNN with a custom dataflow 
#
# Author: Jeremy
#
# Each epoch the CNN training takes a random patch from training and validation images to minimize overfitting
# History: used CNN_smallpics_v1.py with Validate = false, seeds: 42,47,49
#          Started with model from this run of CNN_smallpicsFalse/weights.465.hdf5
import numpy as np
import pandas as pd
import os
from PIL import Image
from skimage.transform import resize
from random import shuffle, seed, randint
from tensorflow import set_random_seed
from keras.applications import imagenet_utils,densenet,inception_resnet_v2,resnet50,inception_v3,mobilenet
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Reshape
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import activations
import sys

if not(4 <= len(sys.argv) <= 5):
   print("Usage: python",sys.argv[0],"MODEL SPLIT VERSION [STARTING_MODEL]")
   sys.exit(-1)

MODELS = {
  'densenet121': {
       'size': 224,
       'preprocessor': densenet.preprocess_input,
  },
  'densenet169': {
       'size': 224,
       'preprocessor': densenet.preprocess_input,
  },
  'densenet201': {
       'size': 224,
       'preprocessor': densenet.preprocess_input,
  },
  'inceptionresnet': {
       'size': 299,
       'preprocessor': inception_resnet_v2.preprocess_input,
  },
  'inception': {
       'size': 299,
       'preprocessor': inception_v3.preprocess_input,
  },
  'mobilenet': {
       'size': 224,
       'preprocessor': mobilenet.preprocess_input,
  },
  'resnet': {
       'size': 224,
       'preprocessor': resnet50.preprocess_input,
  },
}
MODEL = sys.argv[1]
if MODEL not in MODELS.keys():
    print("Bad model argument:", MODEL)
    sys.exit(-1)

#Either need to >>>>manually split training set into training and validation<<<<
#  OR need to make txt files of the image names (this will be the validation set)
VALID_SPLIT = int(sys.argv[2])
if VALID_SPLIT not in range(3):
    print("Bad model argument:", MODEL)
    sys.exit(-1)
VERSION=sys.argv[3]

SCRIPT_NAME="unalt_" + MODEL + "_split" + str(VALID_SPLIT) + "_v" + VERSION
if os.path.exists(SCRIPT_NAME + "/"):
   print("Directory for saved models already exists:", SCRIPT_NAME)
   sys.exit(-1)

STARTING_MODEL=""
if len(sys.argv) == 5:
   STARTING_MODEL=sys.argv[4]

PATH="../../.." # path to training files
DIR_SEP = "/" # "/" for unix, "\\" for windows
SEEDS = [randint(0,10000),randint(0,10000),randint(0,10000)]

print(SCRIPT_NAME)
print("Model:",MODEL)
print("Validation split:", VALID_SPLIT)
print("Starting Model:", STARTING_MODEL)
print("Seeds:", SEEDS)


# Set random seed so that the results are repeatable
seed(SEEDS[0])
np.random.seed(SEEDS[1])
set_random_seed(SEEDS[2])

# ## Generate Training and Validation splits
#
# So that they are compatible with the custom dataflow generator

# Any results you write to the current directory are saved as output.
list_paths = []
for subdir, dirs, files in os.walk(PATH + DIR_SEP + "input"):
    for file in files:
        filepath = subdir + DIR_SEP + file
        list_paths.append(filepath)



######  CHANGE THESE TO ROCK & CLASSICAL
list_classes = ['Sony-NEX-7',
 'Motorola-X',
 'HTC-1-M7',
 'Samsung-Galaxy-Note3',
 'Motorola-Droid-Maxx',
 'iPhone-4s',
 'iPhone-6',
 'LG-Nexus-5x',
 'Samsung-Galaxy-S4',
 'Motorola-Nexus-6']
dict_classes = {'Sony-NEX-7':0,
 'Motorola-X':1,
 'HTC-1-M7':2,
 'Samsung-Galaxy-Note3':3,
 'Motorola-Droid-Maxx':4,
 'iPhone-4s':5,
 'iPhone-6':6,
 'LG-Nexus-5x':7,
 'Samsung-Galaxy-S4':8,
 'Motorola-Nexus-6':9}

def get_class_from_path(filepath):
    return os.path.dirname(filepath).split(os.sep)[-1]

def label_transform(label):
    return dict_classes[label]



 ##############################################################################################


#Take a potentially random patch of image
preprocess_input = MODELS[MODEL]['preprocessor']
def read_and_crop(filepath, left=None, top=None, random = True, margin = 0, width = 112, height = 112):
    im_array = np.array(Image.open((filepath)), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    if left == None:
        if random:
            left = randint(margin, pil_im.size[0] - margin - width + 1)
        else:
            left = (pil_im.size[0] - width) // 2
    if top == None:
        if random:
            top = randint(margin, pil_im.size[1] - margin - height + 1)
        else:
            top = (pil_im.size[1] - height) // 2
    new_array = np.array(pil_im.crop((left,top,left+width,top+height)))
    return preprocess_input(new_array/1.0)


 ##############################################################################################







# Grab file list and labels
list_train = [filepath for filepath in list_paths if DIR_SEP + "train" + DIR_SEP in filepath]
train_ex1 = [[],[],[],[],[],[],[],[],[],[]]
train_ex2 = [[],[],[],[],[],[],[],[],[],[]]
for filepath in list_train:
    label = label_transform(get_class_from_path(filepath))
    train_ex1[label].append(filepath)
for split in range(3):
    if split != VALID_SPLIT:
        for line in open("./level2_split" +str(split),'r'):
            filepath = PATH + DIR_SEP + line.strip()
            label = label_transform(get_class_from_path(filepath))
            train_ex2[label].append(filepath)

list_valid = [PATH + DIR_SEP + line.strip() for line in open("./level2_split" + str(VALID_SPLIT),'r')]
valid_ex = [[],[],[],[],[],[],[],[],[],[]]
for filepath in list_valid:
    label = label_transform(get_class_from_path(filepath))
    valid_ex[label].append(filepath)


partition = {'train': [train_ex1,train_ex2,train_ex2], 'validation': [valid_ex]}


print("done assembling training and validation sets")






##########################################################################################################################


# ## Custom Dataflow Generator
#
#
# Code adapted from blog at: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html


##################################################
## USE KERAS DATAGENERATOR (ImageDataGenerator) ##
##################################################
from random import randint

class DataGenerator(object):

    def __init__(self, dim_x, dim_y, dim_z = 3, batch_size = 40, margin=100, random_location = True,batches_per_epoch = 100,nclass=10):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.random_location = random_location
        self.margin = margin
        self.nclass = nclass

    def generate(self, list_IDs):
        # Generates batches of samples
        # Infinite loop
        while 1:
            # Generate batches
            imax = self.batches_per_epoch
            for i in range(imax):
                # Generate data
                X, y = self.__data_generation(list_IDs)

                yield X, y


    def __data_generation(self, list_IDs):
        #Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size), dtype = int)

        for i in range(self.batch_size):
            sector = randint(0,len(list_IDs)-1)
            label = randint(0,self.nclass-1)
            pic_ndx = randint(0,len(list_IDs[sector][label])-1)
            X[i, :, :, :] = read_and_crop(list_IDs[sector][label][pic_ndx],
                                              margin=self.margin,random=self.random_location,
                                              height=self.dim_y,width=self.dim_x)
            y[i] = label

        return X, sparsify(y)


     ## dont need this (used by data generator)
def sparsify(y):
    # Returns labels in binary NumPy array'
    n_classes = 10
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])







#################################################################################################################################################

# ## Train the CNN
#
# Code adapted from: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

# In[10]:

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
from keras.models import Model
nclass = 10
def get_model():

    num_classes = 10

    input_shape=(MODELS[MODEL]['size'],MODELS[MODEL]['size'],3)
    #preprocess = imagenet_utils.preprocess_input

    input_image = Input(shape=input_shape)

    if MODEL == "densenet121":
        base_model = densenet.DenseNet121(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "densenet169":
        base_model = densenet.DenseNet169(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "densenet201":
        base_model = densenet.DenseNet201(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "inceptionresnet":
        base_model = inception_resnet_v2.InceptionResNetV2(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "inception":
        base_model = inception_v3.InceptionV3(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "mobilenet":
        base_model = mobilenet.MobileNet(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "resnet":
        base_model = resnet50.ResNet50(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "vgg16":
        base_model = vgg16.VGG16(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    elif MODEL == "vgg19":
        base_model = vgg19.VGG19(include_top=False, pooling=None, weights='imagenet',input_shape=input_shape)
    else:
        print("Bad model type:",MODEL);
        sys.exit(-1);

    x = input_image
    x = base_model(x) #resent or densenet or whatever
    x = Reshape((-1,))(x)
    #x = Dropout(rate=?)(x)   ## should use this
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.3,         name='dropout_fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.3,         name='dropout_fc2')(x)
    prediction = Dense(nclass, activation ="softmax", name="predictions")(x)

    # this is the model we will train
    my_model = Model(inputs=(input_image), outputs=prediction)

    # first: train only the top layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    opt = optimizers.Adam(lr=1e-4)
    my_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    my_model.summary()
    return my_model









########################################################################################################


print("Getting model")
model = get_model()
print("done")

if not os.path.exists(SCRIPT_NAME + "/"):
    os.makedirs(SCRIPT_NAME + "/")
file_path=SCRIPT_NAME + "/weights.{epoch:04d}.hdf5"

callbacks_list = [ ModelCheckpoint(file_path, monitor='val_acc', verbose=1) ]


# Might need to change this to fit keras data generator
# Parameters
paramsTrain = {'dim_x': MODELS[MODEL]['size'],
          'dim_y': MODELS[MODEL]['size'],
          'dim_z': 3,
          'batch_size': 32,
          'batches_per_epoch': 150,
          'nclass': 10,
          'margin': 100,
          'random_location': True}
paramsValid = {'dim_x': MODELS[MODEL]['size'],
          'dim_y': MODELS[MODEL]['size'],
          'dim_z': 3,
          'batch_size': 32,
          'batches_per_epoch': 80,
          'nclass': 10,
          'margin': 100,
          'random_location': True}


print("starting training")
if STARTING_MODEL != "":
    model.load_weights(STARTING_MODEL)


## Need to change to match ImageDataGenerator from keras
# Generators
training_generator = DataGenerator(**paramsTrain).generate(partition['train'])
validation_generator = DataGenerator(**paramsValid).generate(partition['validation'])

# Train model on dataset
history = model.fit_generator(generator = training_generator,
                        steps_per_epoch = paramsTrain['batches_per_epoch'],
                        validation_data = validation_generator,
                        validation_steps = paramsValid['batches_per_epoch'],
                        epochs=1000,
                        verbose=2,callbacks=callbacks_list)

print(history)



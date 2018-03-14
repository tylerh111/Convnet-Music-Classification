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
from random import shuffle, seed
from tensorflow import set_random_seed


SCRIPT_NAME="CNN_smallpics_v4"
PATH="../../.." # path to training files
DIR_SEP = "/" # "/" for unix, "\\" for windows
TRAIN = True  # if you wish to train model
SEEDS = [42,47,49]
STARTING_MODEL = "CNN_smallpicsFalse/weights.465.hdf5"

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


#Take a potentially random patch of image
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
    return new_array /255

# Grab file list and labels
list_train = [filepath for filepath in list_paths if DIR_SEP + "train" + DIR_SEP in filepath]
train_ex = [[],[],[],[],[],[],[],[],[],[]]
for filepath in list_train:
    label = label_transform(get_class_from_path(filepath))
    train_ex[label].append(filepath)

list_valid = [filepath for filepath in list_paths if DIR_SEP + "valid" + DIR_SEP in filepath]
valid_ex = [[],[],[],[],[],[],[],[],[],[]]
for filepath in list_train:
    label = label_transform(get_class_from_path(filepath))
    valid_ex[label].append(filepath)

    
partition = {'train': train_ex, 'validation': valid_ex}


# ## Custom Dataflow Generator 
# 
# 
# Code adapted from blog at: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html


from random import randint

class DataGenerator(object):

    def __init__(self, dim_x = 112, dim_y = 112, dim_z = 3, batch_size = 40, margin=100, random_location = True,batches_per_epoch = 100,nclass=10):
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
            label = randint(0,self.nclass-1)
            pic_ndx = randint(0,len(list_IDs[label])-1)
            X[i, :, :, :] = read_and_crop(list_IDs[label][pic_ndx],
                                              margin=self.margin,random=self.random_location,
                                              height=self.dim_y,width=self.dim_x)
            y[i] = label

        return X, sparsify(y)


def sparsify(y):
    # Returns labels in binary NumPy array'
    n_classes = 10
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])




# ## Train the CNN
# 
# Code adapted from: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

# In[10]:

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
input_shape = (112,112,3)
nclass = 10
def get_model():

    nclass = 10
    inp = Input(shape=input_shape)
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(norm_inp)
    img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(32, kernel_size=3, activation=activations.relu, padding="same")(img_1)
    img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    img_1 = Convolution2D(64, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    img_1 = Convolution2D(20, kernel_size=2, activation=activations.relu, padding="same")(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    dense_1 = Dense(20, activation=activations.relu)(img_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


# In[ ]:

model = get_model()

if not os.path.exists(SCRIPT_NAME + "/"):
    os.makedirs(SCRIPT_NAME + "/")
file_path=SCRIPT_NAME + "/weights.{epoch:04d}.hdf5"

callbacks_list = [ ModelCheckpoint(file_path, monitor='val_acc', verbose=1) ]

# Parameters
paramsTrain = {'dim_x': 112,
          'dim_y': 112,
          'dim_z': 3,
          'batch_size': 32,
          'batches_per_epoch': 50,
          'nclass': 10,
          'margin': 100,
          'random_location': True}
paramsValid = {'dim_x': 112,
          'dim_y': 112,
          'dim_z': 3,
          'batch_size': 32,
          'batches_per_epoch': 4,
          'nclass': 10,
          'margin': 100,
          'random_location': True}


if TRAIN:
    if STARTING_MODEL != "":
        model.load_weights(STARTING_MODEL)
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

    

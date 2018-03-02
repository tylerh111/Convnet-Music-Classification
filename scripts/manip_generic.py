import numpy as np
import pandas as pd
import os
import sys

from PIL import Image
from skimage.transform import resize
from random import shuffle, seed, choice, randint
from tensorflow import set_random_seed
from io import BytesIO
import jpeg4py as jpeg
import skimage
from skimage import io,exposure
from scipy import misc

from keras.applications import imagenet_utils,densenet,inception_resnet_v2,resnet50,inception_v3,mobilenet
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import activations
from keras.applications import imagenet_utils


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
VALID_SPLIT = int(sys.argv[2])
if VALID_SPLIT not in range(3):
    print("Bad model argument:", MODEL)
    sys.exit(-1)
VERSION=sys.argv[3]

SCRIPT_NAME="manip_" + MODEL + "_split" + str(VALID_SPLIT) + "_v" + VERSION
if os.path.exists(SCRIPT_NAME + "/"):
   print("Directory for saved models already exists:", SCRIPT_NAME)
   sys.exit(-1)

STARTING_MODEL=""
if len(sys.argv) == 5:
   STARTING_MODEL=sys.argv[4]

PATH="../.." # path to training files
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

# Code from Andres Torrubia
MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']
TRANSFORMATIONS = ['same', 'flip_h', 'flip_v', 'flip_d_r', 'flip_d_l', 'rot90', 'rot180', 'rot270']
load_img_fast_jpg  = lambda img_path: jpeg.JPEG(img_path).decode()
load_img  = lambda img_path: np.array(Image.open(img_path))

preprocess_input = MODELS[MODEL]['preprocessor']


def transform_image(img, transformation=None):
   if transformation == None:
      transformation = choice(TRANSFORMATIONS)

   if transformation.startswith('flip'):
      if   transformation[5:] == 'h':   img = np.flipud(img) # y = 0
      elif transformation[5:] == 'v':   img = np.fliplr(img) # x = 0
      elif transformation[5:] == 'd_l': img = np.swapaxes(img,0,1) # y = x
      elif transformation[5:] == 'd_r': img = np.swapaxes(np.fliplr(np.flipud(img)),0,1) # y = -x
   elif transformation.startswith('rot'):
      angle = int(transformation[3:])
      img = np.rot90(img)
      if angle > 90:  img == np.rot90(img)
      if angle > 180: img == np.rot90(img)

   return img


def random_manipulation(img, width, height, manipulation=None):

    if manipulation == None:
        manipulation = choice(MANIPULATIONS)

    #kaggel manipulations
    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        img = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        img = exposure.adjust_gamma(img, gamma)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        img = misc.imresize(img, scale, interp='bicubic')
    else:
        assert False

    #transform image
    img = transform_image(img)

    pil_im = Image.fromarray(img)

    img = np.array(pil_im.crop((0,0,width,height)))
    return preprocess_input(img/1.0)


#Take a potentially random patch of image
def read_and_crop(filepath, left=None, top=None, random = True, margin = 0, width = 112, height = 112):
    height = height * 2
    width = width * 2
    im_array = np.array(Image.open((filepath)), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    if left == None:
        if random:
            if margin > pil_im.size[0] - margin - width + 1:
                return False, None
            left = randint(margin, pil_im.size[0] - margin - width + 1)
        else:
            left = (pil_im.size[0] - width) // 2
    if top == None:
        if random:
            if margin > pil_im.size[1] - margin - height + 1:
                return False, None
            top = randint(margin, pil_im.size[1] - margin - height + 1)
        else:
            top = (pil_im.size[1] - height) // 2
    new_array = np.array(pil_im.crop((left,top,left+width,top+height)))
    return True,new_array

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

# ## Custom Dataflow Generator
#
#
# Code adapted from blog at: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html



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
            res = False
            while res == False:
                sector = randint(0,len(list_IDs)-1)
                label = randint(0,self.nclass-1)
                pic_ndx = randint(0,len(list_IDs[sector][label])-1)
                res, img = read_and_crop(list_IDs[sector][label][pic_ndx],
                                         margin=self.margin,random=self.random_location,
                                         height=self.dim_y,width=self.dim_x)
            X[i, :, :, :] = random_manipulation(img,width=self.dim_x, height=self.dim_y)
            y[i] = label

        return X, sparsify(y)


def sparsify(y):
    # Returns labels in binary NumPy array'
    n_classes = 10
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])




# ## Train the CNN
# 

# In[10]:

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Reshape, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalMaxPool2D, Concatenate
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
    x = base_model(x)
    x = Reshape((-1,))(x)
    #x = Dropout(rate=?)(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.3,         name='dropout_fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.3,         name='dropout_fc2')(x)
    prediction = Dense(nclass, activation ="softmax", name="predictions")(x)

    # this is the model we will train
    my_model = Model(inputs=(input_image), outputs=prediction)

    # compile the model (should be done *after* setting layers to non-trainable)
    opt = optimizers.Adam(lr=1e-4)
    my_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    my_model.summary()
    return my_model



# In[ ]:

model = get_model()

if not os.path.exists(SCRIPT_NAME + "/"):
    os.makedirs(SCRIPT_NAME + "/")
file_path=SCRIPT_NAME + "/weights.{epoch:04d}.hdf5"

callbacks_list = [ ModelCheckpoint(file_path, monitor='val_acc', verbose=1) ]

# Parameters
paramsTrain = {'dim_x': MODELS[MODEL]['size'],
               'dim_y': MODELS[MODEL]['size'],
               'dim_z': 3,
               'batch_size': 3,
               'batches_per_epoch': 150,
               'nclass': 10,
               'margin': 100,
               'random_location': True}
paramsValid = {'dim_x': MODELS[MODEL]['size'],
               'dim_y': MODELS[MODEL]['size'],
               'dim_z': 3,
               'batch_size': 3,
               'batches_per_epoch': 80,
               'nclass': 10,
               'margin': 100,
               'random_location': True}


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
                              verbose=2,
                              callbacks=callbacks_list)

print(history)









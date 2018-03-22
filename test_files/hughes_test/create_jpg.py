
from PIL import Image
import os
import sys

import random
import numpy as np

print('======================================================================')

### Converts png to jpg
PATH = "/media/tdh5188/easystore/data/convnet_input"
PNG_PATH = 'input_png'
JPG_PATH = 'input'

list_paths = []
for subdir,dirs,files in os.walk(PATH + '/' + PNG_PATH):
	for file in files:
		filepath = subdir + '/' + file
		list_paths.append(filepath)


#def convert_path(path, r_l):
#    #returns ".../data/convnet_input/input/Classical/~~~~~_left.jpg
#	return (PATH + '/' + JPG_PATH + '/' + str(os.path.dirname(path).split(os.sep)[-1]) + '/' + os.path.basename(path)[:-4] + '_left.jpg',
#			PATH + '/' + JPG_PATH + '/' + str(os.path.dirname(path).split(os.sep)[-1]) + '/' + os.path.basename(path)[:-4] + '_right.jpg')

#for filepath in list_paths:
# 		label = label_transform(get_class_from_path(filepath))
# 		# randomly decide to add to training set or validation set
# 		if random.uniform(0,1) >= 0.10:
# 			train_set[label].append((filepath, label))
# 		else:
# 			valid_set[label].append((filepath, label))


#split into training and validation set and convert file
def convert_path(path):
	left = right = PATH + '/' + JPG_PATH + '/'
	rop = str(os.path.dirname(path).split(os.sep)[-1]) + '/' + os.path.basename(path)[:-4]

	if random.uniform(0,1) >= 0.12:
		left += 'train/' + rop + '_left.jpg'
	else:
		left += 'valid/' + rop + '_left.jpg'

	if random.uniform(0,1) >= 0.12:
		right += 'train/' + rop + '_right.jpg'
	else:
		right += 'valid/' + rop + '_right.jpg'

	return left,right



# for file in list_paths:
# 	print('\tConverting from png to jpg:', file)
# 	im = Image.open(file)
# 	im = im.convert('RGB')
# 	im.save(convert_path(file))

for file in list_paths:
	print('\tConverting from png to jpg:',file)
	new_paths = convert_path(file)
	img = Image.open(file).convert('RGB')
	img.crop((0,0,img.size[0],(img.size[1] // 2))).save(new_paths[0]) #left channel (top)
	img.crop((0,(img.size[1] // 2),img.size[0],img.size[1])).save(new_paths[1]) #right channel (bottom)


#i = 0
#
#for file in list_paths:
#    if i > 0: break
#    i+=1
#    print('\tConverting from png to jpg:',file)
#    new_paths = convert_path(file)
#    img = Image.open(file).convert('RGB')
#    img.crop((0,0,img.size[0],(img.size[1] // 2))).save(new_paths[0]) #left channel (top)
#    img.crop((0,(img.size[1] // 2),img.size[0],img.size[1])).save(new_paths[1]) #right channel (bottom)



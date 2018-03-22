
from PIL import Image
import os
import sys

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


def convert_path(path):
	return PATH+'/'+JPG_PATH+'/'+str(os.path.dirname(path).split(os.sep)[-1])+'/'+os.path.basename(path)


for file in list_paths:
	print('\tConverting from png to jpg:', file)
	im = Image.open(file)
	im = im.convert('RGB')
	im.save(convert_path(file))

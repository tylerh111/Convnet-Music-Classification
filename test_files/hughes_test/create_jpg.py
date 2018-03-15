
from PIL import Image
import os
import sys

### Converts png to jpg
PATH="/media/tdh5188/easystore/convnet_input"

list_paths = []
for subdir,dirs,files in os.walk(PATH + "/input_png"):
	for file in files:
		filepath = subdir + "/" + file
		list_paths.append(filepath)


def convert_path (path):
	return PATH+'/input/'+str(os.path.dirname(path).split(os.sep)[-1])+'/'+os.path.basename(path)


for file in list_paths:
	im = Image.open(file)
	im = im.convert('RGB')
	im.save(convert_path(file))

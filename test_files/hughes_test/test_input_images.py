

import numpy as np
from PIL import Image


filepath_png = "/media/tdh5188/easystore/convnet_input/input/Classical/05_-_Mozart,_Wolfgang_Amadeus_-_Symphonie_Nr.17_in_G-dur,_KV_129-_2._Andante.png"

#print("\n\nfilepath =", filepath_png)
#im_array = np.array(Image.open(filepath_png),dtype = "uint8")
#pil_im = Image.fromarray(im_array)
#print("im_array:\n",im_array)
#print("\npil_im:\n",pil_im)

#print("\nAFTER:")
#rgb_im = pil_im.convert('RGB')

#rgb_im.save('test.jpg')

#print("array :\n",np.array(rgb_im))
#print("\nrgb_im:\n",rgb_im)


#rgb_im.show()


im = Image.open(filepath_png)
print(im.mode)
im = im.convert('RGB')
print(im.mode)
im.save('test.jpg')




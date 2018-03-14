
import soundfile as sf
import numpy as np

import matplotlib.pyplot as plt

PATH = '../audiofiles/'
FILE = '01 - Symphonic Variations.flac'


arr = sf.read(PATH + FILE)

print(arr)
print(arr[0])
print(arr[1])

lenarr = len(arr[0])
print('len =', lenarr)

print()

narr = np.array(arr[0])

narr2 = np.array([])
x_narr = np.array([])
y_narr = np.array([])

i = -1
for [x,y] in narr:
    i+=1
    if (i % 100000) == 0:
        np.append(x_narr, x)
        np.append(y_narr, y)

print('\ntime to plot\n')

plt.scatter(x_narr, y_narr)
plt.show()











#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 02:05:19 2018

@author: tdh5188
"""

import numpy as np

def enframe(x,S,L):
   # w = 0.54*np.ones(L)
    #for n in range(0,L):
     #   w[n] = w[n] - 0.46*math.cos(2*math.pi*n/(L-1))
    print('\ninside enframe')
    print('x =',x)
    print('S =',S)
    print('L =',L)
    w = np.hamming(L)
    print('w =',w)
    frames = []
    nframes = 1+int((len(x)-L)/S)
    print('nframes =',nframes)
    for t in range(0,nframes):
        if t == 0:
            print('t =', t)
            print('(t*S) =',(t*S))
            print('(t*S+L) =',(t*S+L))
            print('x[(t*S):(t*S+L)] =', (x[(t*S):(t*S+L)]))
            temp = np.copy(x[(t*S):(t*S+L)])*w
            print('temp =',temp)
        frames.append(np.copy(x[(t*S):(t*S+L)])*w)
        #frames.append(temp)
    return(frames)

#w = np.hanning(200)
#f2 = plt.figure(figsize=(14,4))
#plt.plot(w)




print('starting')

w_frames = enframe(w_wav,int(0.01*w_fs),int(0.035*w_fs))
plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(np.linspace(0,0.005,len(w_frames[11])),w_frames[11])
plt.title('A frame from White Fang')






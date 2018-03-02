
import os
from subprocess import check_call
from tempfile import mktemp
from scikits.audiolab import flacread, play
from scipy.signal import remez, lfilter
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np


# convert mp3, read wav
#mp3filename = 'XC124158.mp3'
flacfile = '/home/tdh5188/workspace/comp596/Convnet-Music-Classification/03-Time.flac'
#wname = mktemp('.wav')



#print(wname)

#check_call(['avconv', '-i', flacfile, wname])
#sig, fs, enc = flacread(wname)
sig, fs, enc = flacread(flacfile)


#os.unlink(wname)






# bandpass filter
bands = np.array([0,3500,4000,5500,6000,fs/2.0]) / fs
desired = [0, 1, 0]
b = remez(513, bands, desired)
sig_filt = lfilter(b, 1, sig)
sig_filt /=  1.05 * max(abs(sig_filt)) # normalize

subplot(211)
specgram(sig, Fs=fs, NFFT=1024, noverlap=0)
axis('tight'); axis(ymax=8000)
title('Original')
subplot(212)
specgram(sig_filt, Fs=fs, NFFT=1024, noverlap=0)
axis('tight'); axis(ymax=8000)
title('Filtered')
show()

play(sig_filt, fs)



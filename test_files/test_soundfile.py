#### TUTORIAL FROM: https://courses.physics.illinois.edu/ece590sip/sp2018/spectrograms1_wideband_narrowband.html
#### currently trying to adapt for flac files


import io
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import urllib.request as request

import soundfile as sf


#data, samplerate = sf.read('03-Time.flac')






'''Plotting a waveform'''
print('plotting waveform')
print('getting input files from web')

# First, "Pride and Prejudice".  I found the following URL by going to gutenberg.org, searching for "Pride and Prejudice," 
# clicking on the audio version, then copying the URL from the first "OGG Vorbis Audio" file
p_url = 'http://www.gutenberg.org/files/26301/ogg/26301-01.ogg'
p_data, p_fs = sf.read(io.BytesIO(request.urlopen(p_url).read()))
# Now, "White Fang".  
w_url = 'http://www.gutenberg.org/files/23976/ogg/23976-01.ogg'
w_data, w_fs = sf.read(io.BytesIO(request.urlopen(w_url).read())) 
# Let's save both of those files in the current directory, so we can read them using other programs
# Let's save them as WAV; it's a big file, but can be opened by a wider variety of programs
sf.write('pride_and_prejudice_part1.wav',p_data,p_fs)
sf.write('white_fang_part1.wav',w_data,w_fs)




print('preprocessing songs')

# Afer opening each file in Praat, we discover that
# the speech part of "Pride and Prejudice" starts at about t=0.74s 
# and the speech part of "White Fang" starts at about t=0.42s
# Let's cut out an 0.6s segment from each, and plot them
p_wav = p_data[int(0.74*p_fs):int((0.74+0.6)*p_fs)]
w_wav = w_data[int(0.42*w_fs):int((0.42+0.6)*w_fs)]  #############################
# We'll use the numpy function "linspace" to create a time axis for plotting
p_timeaxis = np.linspace(0,0.6,len(p_wav))
w_timeaxis = np.linspace(0,0.6,len(w_wav))

print('plotting')
# And plot them
f1=plt.figure(1,figsize=(14,4))
plt.subplot(211)
plt.plot(p_timeaxis,p_wav)
plt.title('First segment from Pride and Prejudice')
plt.subplot(212)
plt.plot(w_timeaxis,w_wav)
plt.title('First segment from White Fang')





'''CREATING A SPECTROGRAM'''

'''1) Enframe the audio'''

print('\nCREATING A SPECTROGRAM\n')
print('1) Enframe the audio')
print('defining function enframe')

def enframe(x,S,L):
   # w = 0.54*np.ones(L)
    #for n in range(0,L):
     #   w[n] = w[n] - 0.46*math.cos(2*math.pi*n/(L-1))
    w = np.hamming(L)
    frames = []
    nframes = 1+int((len(x)-L)/S)
    for t in range(0,nframes):
        frames.append(np.copy(x[(t*S):(t*S+L)])*w)
    return(frames)

w = np.hanning(200)
f2 = plt.figure(figsize=(14,4))
plt.plot(w)




print('plotting frame 12 of each song')

w_frames = enframe(w_wav,int(0.01*w_fs),int(0.035*w_fs))
p_frames = enframe(p_wav,int(0.01*p_fs),int(0.035*p_fs))
plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(np.linspace(0,0.005,len(w_frames[11])),w_frames[11])
plt.title('A frame from White Fang')
plt.subplot(212)
plt.plot(np.linspace(0,0.005,len(p_frames[11])),p_frames[11])
plt.title('A frame from Pride and Prejudice')


'''2) Create STFT from the frames'''

print('\n2) Create STFT from the frames')
print('defining function stft')

def stft(frames,N,Fs):
    stft_frames = [ fftpack.fft(x,N) for x in frames]
    freq_axis = np.linspace(0,Fs,N)
    return(stft_frames, freq_axis)


print('testing function')

(w_stft, w_freqaxis) = stft(w_frames, 1024, w_fs)
(p_stft, p_freqaxis) = stft(p_frames, 1024, p_fs)
plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(w_freqaxis,np.log(np.maximum(1,abs(w_stft[11])**2)))
plt.ylabel('Magnitude Squared STFT')
plt.title('Spectrum of a frame from White Fang')
plt.subplot(212)
plt.plot(p_freqaxis,abs(p_stft[11])**2)
plt.title('Spectrum of a frame from Pride and Prejudice')
plt.xlabel('Frequency (Hertz)')



print('preprocessing on plots')

plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(w_freqaxis[w_freqaxis<=5000],np.log(abs(w_stft[11][w_freqaxis<=5000])))
plt.title('White Fang')
plt.ylabel('Magnitude STFT')
plt.subplot(212)
plt.plot(p_freqaxis[p_freqaxis<=5000],np.log(abs(p_stft[11][p_freqaxis<=5000])))
plt.title('Pride and Prejudice')
plt.xlabel('Frequency (Hertz)')


'''3) Compute Level'''

print('\n3) Compute Level')
print('defining function stft2level')


def stft2level(stft_spectra,max_freq_bin):
    magnitude_spectra = [ abs(x) for x in stft_spectra ]
    max_magnitude = max([ max(x) for x in magnitude_spectra ])
    min_magnitude = max_magnitude / 1000.0
    for t in range(0,len(magnitude_spectra)):
        for k in range(0,len(magnitude_spectra[t])):
            magnitude_spectra[t][k] /= min_magnitude
            if magnitude_spectra[t][k] < 1:
                magnitude_spectra[t][k] = 1
    level_spectra = [ 20*np.log10(x[0:max_freq_bin]) for x in magnitude_spectra ]
    return(level_spectra)


print('testing function')

max_freq = 5000 # choose 5000Hz as the maximum displayed frequency
p_sgram = stft2level(p_stft, int(1024*max_freq/p_fs))
w_sgram = stft2level(w_stft,int(1024*max_freq/w_fs))
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(p_sgram)),origin='lower',extent=(0,600,0,max_freq),aspect='auto')
plt.title('Spectrogram of a 600ms segment of Pride and Prejudice')
plt.xlabel('Time (ms)')
plt.ylabel('Freq (Hz)')
print(p_fs)


'''Spectrogram Function'''

print('\nSpectrogram Function')
print('defining function sgram')


def sgram(x,frame_skip,frame_length,fft_length, fs, max_freq):
    frames = enframe(x,frame_skip,frame_length)
    (spectra, freq_axis) = stft(frames, fft_length, fs)
    sgram = stft2level(spectra, int(max_freq*fft_length/fs))
    max_time = len(frames)*frame_skip/fs
    return(sgram, max_time, max_freq)


print('computing same spectrogram as above')

(p_sgram, p_maxtime, p_maxfreq) = sgram(p_wav, int(0.002*p_fs), int(0.035*p_fs), 1024, p_fs, 5000)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(p_sgram)),origin='lower',extent=(0,p_maxtime,0,p_maxfreq),aspect='auto')



'''Wideband/Narrowband spectrogram'''
print('\nWideband/Narrowband spectrogram')

print('wideband spectrogram for "Pride and Prejudice"')
(p_sgram,p_maxtime, p_maxfreq) = sgram(p_wav, int(0.001*p_fs), int(0.004*p_fs), 1024, p_fs, 5000)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(p_sgram)),origin='lower',extent=(0,p_maxtime,0,p_maxfreq),aspect='auto')
plt.title('Wideband Spectrogram of a segment from Pride and Prejudice')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')



print('wideband spectrogram for "White Fang"')
(w_sgram, w_maxtime, w_maxfreq) = sgram(w_wav, int(0.001*w_fs), int(0.004*w_fs), 1024, w_fs, 5000)  
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(w_sgram)),origin='lower',extent=(0,w_maxtime,0,w_maxfreq),aspect='auto')
plt.title('Wideband Spectrogram of a segment from White Fang')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')



print('narrowband spectrogram for "White Fang"')
(narrowband_w_sgram,w_maxtime,w_maxfreq) = sgram(w_wav, int(0.001*w_fs), int(0.035*w_fs), 1024, w_fs, 5000)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(narrowband_w_sgram)),origin='lower',extent=(0,600,0,5000),aspect='auto')
plt.title('Narrowband Spectrogram of a segment from White Fang')















import io
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import urllib.request as request

import soundfile as sf


#data, samplerate = sf.read('03-Time.flac')






'''Plotting a waveform'''
print('Plotting Waveform')
#print('getting input files from web')


audio_file_name = '03-Time'
audio_file_type = '.flac'

data, fs = sf.read(audio_file_name + audio_file_type)

sf.write(audio_file_name + '.wav', data, fs)


print('preprocessing songs')

# Afer opening each file in Praat, we discover that
# the speech part of "Pride and Prejudice" starts at about t=0.74s 
# and the speech part of "White Fang" starts at about t=0.42s
# Let's cut out an 0.6s segment from each, and plot them
wav = data[int(0.74*fs):int((0.74+0.6)*fs)]

# We'll use the numpy function "linspace" to create a time axis for plotting
timeaxis = np.linspace(0,0.6,len(wav))


print('plotting')
# And plot them
f1=plt.figure(1,figsize=(14,4))
plt.subplot(211)
plt.plot(timeaxis, wav)
plt.title('First segment from ' + audio_file_name + audio_file_type)





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

#w = np.hanning(200)
#f2 = plt.figure(figsize=(14,4))
#plt.plot(w)


######## PROBLEM:: wav is a 2D array, we need 1D array <<<<<<<<<<<

print('plotting frame 12 of each song')

frames = enframe(wav,int(0.01*fs),int(0.05*fs))
plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(np.linspace(0,0.005,len(frames[11])),frames[11])
plt.title('A frame from ' + audio_file_name + audio_file_type)


'''2) Create STFT from the frames'''

print('\n2) Create STFT from the frames')
print('defining function stft')

def stft(frames,N,Fs):
    stft_frames = [ fftpack.fft(x,N) for x in frames]
    freq_axis = np.linspace(0,Fs,N)
    return(stft_frames, freq_axis)


print('testing function')

(stft, freqaxis) = stft(frames, 1024, fs)
plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(freqaxis,np.log(np.maximum(1,abs(stft[11])**2)))
plt.ylabel('Magnitude Squared STFT')
plt.title('Spectrum of a frame from ' + audio_file_name + audio_file_type)
plt.xlabel('Frequency (Hertz)')



print('preprocessing on plots')

plt.figure(figsize=(14,4))
plt.subplot(211)
plt.plot(freqaxis[freqaxis<=5000],np.log(abs(w_stft[11][freqaxis<=5000])))
plt.title(audio_file_name + audio_file_type)
plt.ylabel('Magnitude STFT')




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
sgram = stft2level(stft, int(1024*max_freq/fs))
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(sgram)),origin='lower',extent=(0,600,0,max_freq),aspect='auto')
plt.title('Spectrogram of a 600ms segment of ' + audio_file_name + audio_file_type)
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

(sgram, maxtime, maxfreq) = sgram(wav, int(0.002*fs), int(0.035*fs), 1024, fs, 5000)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(sgram)),origin='lower',extent=(0,maxtime,0,maxfreq),aspect='auto')



'''Wideband/Narrowband spectrogram'''
print('\nWideband/Narrowband spectrogram')

print('wideband spectrogram for ' + audio_file_name + audio_file_type)
(sgram,maxtime, maxfreq) = sgram(wav, int(0.001*fs), int(0.004*fs), 1024, fs, 5000)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(sgram)),origin='lower',extent=(0,maxtime,0,maxfreq),aspect='auto')
plt.title('Wideband Spectrogram of a segment from ' + audio_file_name + audio_file_type)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')



print('narrowband spectrogram for ' + audio_file_name + audio_file_type)
(narrowband_sgram,maxtime,maxfreq) = sgram(wav, int(0.001*fs), int(0.035*fs), 1024, fs, 5000)
plt.figure(figsize=(14,4))
plt.imshow(np.transpose(np.array(narrowband_sgram)),origin='lower',extent=(0,600,0,5000),aspect='auto')
plt.title('Narrowband Spectrogram of a segment from ' + audio_file_name + audio_file_type)















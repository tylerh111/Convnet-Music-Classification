import soundfile as sf
import numpy as np
import random 

song = sf.read("xd.flac")

print(np.shape(song[0].transpose()[0]))

rightChannel = song[0].transpose()[0]

leftChannel = song[0].transpose()[1]

print(rightChannel)

#for i in range(0, len(rightChannel)): #manipulates
#    rightChannel[i] = rightChannel[i] + random.uniform(-10,10)

print(rightChannel)

newR = sf.write("right.flac",rightChannel,song[1])

newL = sf.write("left.flac",leftChannel,song[1])

startingPoint = random.randint(0,len(rightChannel)-150000)

fragmentR = sf.write("fragmentR.flac",rightChannel[startingPoint:startingPoint+150000],song[1] )

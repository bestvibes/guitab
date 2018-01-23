"""
A better way to extract data from audio files to train 
machine learning model with. 

Reference: https://arxiv.org/pdf/1508.01774.pdf; Page 7
Warning: Currently incomplete. VERY HACKY. Clean code ASAP.
"""

import scipy
import pylab
import wave
import struct
import sys
import os
import csv

from scipy import fftpack

def stft(data, cp, do, hop):
    dos = int(do*cp)
    w = scipy.kaiser(dos,12) # 12 is very high for kaiser window
    temp=[]
    wyn=[]
    for i in range(0, len(data)-dos, int(hop * cp)):
        temp2 = scipy.fft(w*data[i:i+dos])
        temp = scipy.fftpack.fftshift(temp2)

        max=-1
        for j in range(0, len(temp),1):
            licz=temp[j].real**2+temp[j].imag**2
            if( licz>max ):
                max = licz
                maxj = j
        wyn.append(maxj)
    #wyn = scipy.array([scipy.fft(w*data[i:i+dos])
        #for i in range(0, len(data)-dos, 1)])
    return wyn

MIDI = list(range(40, 90))
for value in MIDI:
	files = list()
	for file in os.listdir("./Guitar-notes"):
		if "0" + str(value) + "-" in os.path.join("./Guitar-notes", file):
			files.append(os.path.join("./Guitar-notes", file))

	for filename in files:
		if ".wav" not in filename:
			continue

		file = wave.open(filename)
		if file.getnchannels() == 2:
			bity = file.readframes(file.getnframes())[::2]
		else:
			bity = file.readframes(file.getnframes())
		data = struct.unpack('{n}h'.format(n=file.getnframes()), bity)
		file.close()

		cp = 16000 # 16kHz
		do = 0.064 # 64 ms
		hop = 0.032 # 32 ms

		wyn=stft(data,cp,do,hop)

		output = list()
		for i in range(0, len(wyn), 1):
		    if wyn[i] != 0:
		    	output.append(wyn[i])

		if len(output) < 20:
			os.remove(filename)
			continue

		with open("preprocessed.csv", 'a') as csv_file:
		    writer = csv.writer(csv_file, delimiter=',')
		    writer.writerow(output + [value])
		print(value)
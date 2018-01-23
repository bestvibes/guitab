#!/usr/bin/env python
"""
Applies fast fourier transform on a wav file and returns a 2D list
of frequencies used for training/predicting.

Warning: VERY HACKY. Clean code ASAP.
"""

from __future__ import print_function

import struct
import wave
import sys
import csv
import pickle
import os
import glob

import numpy as np

FPS = 25.0

nFFT = 512
BUF_SIZE = 4 * nFFT
SAMPLE_SIZE = 2
CHANNELS = 1
RATE = 16000

numpixels = 60

def animate(i, wf, MAX_y):
	N = (int((i + 1) * RATE / FPS) - wf.tell()) / nFFT
	if not N:
		return line,
	N *= nFFT
	data = wf.readframes(N)


	# Unpack data
	y = np.array(struct.unpack("%dh" % (len(data) / wf.getsampwidth()), data)) / MAX_y

	if wf.getnchannels() == 2:
		y = y[::2]

	Y_L = np.fft.fft(y, nFFT)
	Y = list(abs(Y_L[-nFFT / 2:-1]))

	#return Y
	return [y if y > 0.001 else 0 for y in Y]
	# if Y[0] > 0.001:
	#   #print(Y)
	#   return Y

def fft(file):
	MAX_y = 2.0 ** (SAMPLE_SIZE * 8 - 1)
	wf = wave.open(file, 'rb')
	# assert wf.getnchannels() == CHANNELS
	# assert wf.getsampwidth() == SAMPLE_SIZE
	# assert wf.getframerate() == RATE
	frames = wf.getnframes()

	output = list()

	for i in xrange(int(frames / wf.getframerate() * FPS)):
		frame = animate(i, wf, MAX_y)
		if frame[0] > 0.001:
			output.append(frame)

	wf.close()

	return output

def getData(files, midi):
	for item in files:
		data = fft(item)
		with open("test.csv", 'a') as csv_file:
			writer = csv.writer(csv_file, delimiter=',')
			for item in data:
				writer.writerow(item)

	finaldata = list(csv.reader(open("test.csv")))
	# print(finaldata)
	# print('-' * 79)

	with open(midi, 'a') as f:
		pickle.dump(finaldata,f)

	os.remove("test.csv")

if __name__ == '__main__':
	data = fft("Guitar_Scales_-_C_major_4th_string.wav")
	print(data)
	with open("test.csv", 'a') as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for item in data:
			writer.writerow(item)
	


	# MIDI = list(range(40, 90))
	# for value in MIDI:
	#   files = list()
	#   for file in os.listdir("./Guitar-notes"):
	#     if "0" + str(value) + "-" in os.path.join("./Guitar-notes", file):
	#       files.append(os.path.join("./Guitar-notes", file))

	#   print(files)
	#   print(len(files))
	#   getData(files, "Data/" + str(value))



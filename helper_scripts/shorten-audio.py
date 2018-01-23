"""
A simple python script that shortens all the audio
files in a directory to a specified length.
Relies on third-party package: pydub
"""

from pydub import AudioSegment

import os

DIR_OF_MUS_FILES = "."
NEW_LENGTH_IN_MS = 2500

for curr_file in os.listdir(DIR_OF_MUS_FILES):
	# Only .wav files should be used for training
	if ".wav" not in curr_file:
		continue

	raw_audio = AudioSegment.from_file(curr_file, format="wav")
	shortened_audio = raw_audio[:NEW_LENGTH_IN_MS]
	shortened_audio.export(curr_file, format="wav")
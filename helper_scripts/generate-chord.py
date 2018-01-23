"""
Given a directory of audio files, specify a tab notation
for a chord to generate an audio file of that chord.

Warning: VERY HACKY. Clean code ASAP.
"""

#!/usr/bin/env python

import pydub
import os

# Array showing guitar string's relative pitches, in semi-tones, with "0" being low E
# ["Low E", "A", "D", "G", "B", "High E"]
strings = [40, 45, 50, 55, 60, 65];

def midi_to_label(midi):
    if (midi < strings[0]):
        raise ValueError("Note " + note + " is not playable on a guitar in standard tuning.")

    idealString = 0
    for string, string_midi in enumerate(strings):
        if (midi < string_midi):
            break
        idealString = string

    label = [-1, -1, -1, -1, -1, -1]
    label[idealString] = midi - strings[idealString];
    return label

def label_to_midi(label):
    outp = []
    for string, fret in enumerate(label):
        if (fret != -1):
            outp.append(fret + strings[string])
        else:
            outp.append(-1)
    return outp


if __name__ == '__main__':
	# os.remove("C_chord.wav")

	chords = [[3, 2, 0, 0, 3, 3], # G
			  [-1, 3, 2, 0, 1, 0], # C
			  [0, 2, 2, 1, 0, 0], # E
			  [0, 2, 2, 0, 0, 0], # Em
			  [-1, -1, 0, 3, 2, 3], # D
			  [1, 3, 3, 2, 1, 1], # F
			  [-1, 0, 2, 2, 2, 0], # A
			  [-1, 0, 2, 2, 1, 0], # Am
			  [-1, 3, 2, -1, 3, 3]] # Cadd9

	notes = list(map(label_to_midi, chords))
	for note in notes:
		print(notef())

	# sounds = list()
	# for note in notes:
	# 	if note < 40:
	# 		continue
	# 	elif note > 89:
	# 		continue

	# 	for file in os.listdir("."):
	# 		if "-0" + str(note) + "-" in file:
	# 			sounds.append(file)
	# 			break

	# print(sounds)

	# chord = list()
	# for sound in sounds:
	# 	chord.append(pydub.AudioSegment.from_wav(sound))

	# print(chord)

	# combined_chord = chord[0]
	# for i in range(1, len(chord)):
	# 	combined_chord = combined_chord.overlay(chord[i], position=(100 * i))

	# combined_chord.export("C_chord.wav", format="wav")





	

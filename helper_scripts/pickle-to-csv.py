"""
Given a directory of pickle files where each file is
named after a MIDI number, extract the data from all
files and write to a CSV. Run this script from within
the same directory where all the pickle files are.
"""

import csv
import pickle

FIRST_MIDI_NUM = 40
LAST_MIDI_NUM = 89
OUTPUT_FILE = "data.csv"

midi_nums = list(range(FIRST_MIDI_NUM, LAST_MIDI_NUM + 1)) # Want to include the last midi num

for curr_num in midi_nums:
	# Naming scheme for pickle files was 0xx where
	# xx was the 2 digit midi number.
	# Update if using a different naming scheme
	data_from_pickle = pickle.load(open("0" + str(curr_num)))

	with open(OUTPUT_FILE, 'a') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',')
		for row in data_from_pickle:
			# Last column is midi number label
			csv_writer.writerow(row + [str(curr_num)])
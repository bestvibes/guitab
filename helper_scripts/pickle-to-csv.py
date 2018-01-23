"""
Given a directory of pickle files where each file is
named after a MIDI number, extract the data from all
files and write to a CSV.

Warning: VERY HACKY. Clean code ASAP
"""

import csv
import pickle

if __name__ == '__main__':
	values = list(range(40, 90))
	for value in values:
		input = pickle.load(open("0" + str(value)))
		with open("data.csv", 'a') as csv_file:
			writer = csv.writer(csv_file, delimiter=',')
			for item in input:
				writer.writerow(item + [str(value)])
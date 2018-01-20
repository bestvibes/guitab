import tensorflow as tf
import numpy as np
import cPickle as pickle
import sys 

# Array showing guitar string's relative pitches, in semi-tones, with "0" being low E
# ["Low E", "A", "D", "G", "B", "High E"]
strings = [40, 45, 50, 55, 60, 65];

# our neural network
from nn import nn_train

def midi_to_label(midi):
    if (midi < strings[0]):
        raise Error("Note " + note + " is not playable on a guitar in standard tuning.")

    idealString = 0
    for string, string_midi in enumerate(strings):
        if (midi < string_midi):
            break
        idealString = string

    label = [-1, -1, -1, -1, -1, -1]
    label[idealString] = midi - strings[idealString];
    return label

def process(filename):
    label = midi_to_label(int(filename.split('/')[-1]))

    f = open(filename, 'r')
    data = pickle.load(f)
    nn_train(data, label)
    f.close()

def main():
    process(sys.argv[1])

if(__name__ == "__main__"):
    main()
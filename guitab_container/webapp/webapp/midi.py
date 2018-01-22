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
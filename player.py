#!/usr/bin/python

import numpy as np
import simpleaudio as sa
import threading as th
import time

import matplotlib.pyplot as plt

step_table = {
    'b' : 2,
    'c' : 3,
    'd' : 5,
    'e' : 7,
    'f' : 8,
    'g' : 10,
    'a' : 12
}

sample_notes = [
    [('0c',16),('4e',16),('4g',16)],
    [('0b',16),('0e',16),('0g',16),('4d',16),('8g',16)],
    [('0a',8),('2a#',8),('4c',8),('6c#',8)]
]

sample_notes2 = [
    [('0a', 16)]
]

sample_rate = 44100
note_period = 1.25
finger_num = 10

hrm_amp = [0.17, 0.985, 0.185, 0.110, 0.26, 0.095, 0.080]  # timbre


class SoundGenerator(th.Thread):
    def __init__(self, threadID):
        self.id = threadID

    def run(self):
        # play sound
        pass

class NoteHolder():
    def __init__(self, name):
        self.name = name
        self.note = np.zeros(1)
        self.isEmpty = True

    def popNoteSlice(self):
        if self.timeleft == 0:
            print(self.name, 'is empty!')
            return 0

        note_slice = self.note[:self.maxIdx]
        self.note = self.note[self.maxIdx:]
        self.timeleft -= 1

        if(self.timeleft==0): self.reset()

        print('note slice:', note_slice, len(note_slice))

        return note_slice

    def insertNote(self, note, duration):
        self.note = getNotePlayable(note, hrm_amp, note_period)
        self.note_len = len(self.note)
        self.timeleft = duration
        self.isEmpty = False
        self.maxIdx = self.note_len//16

        print('note:', len(self.note))

    def reset(self):
        self.note = np.zeros(1)
        self.note_len = len(self.note)
        self.isEmpty = True

def getADSR(sample_rate, note_period):
    # sound envelope: env_<comp> = (<duration> , <amplitude>)
    env_a = (0.05, 1.0)
    env_d = (0.1, 0.15)
    env_s = (0.1, env_d[1])
    env_r = (round(note_period - env_a[0] - env_d[0] - env_s[0], 2), 0)

    adsr_t = [env_a[0], env_d[0], env_s[0], env_r[0]]
    adsr_a = [env_a[1], env_d[1], env_s[1], env_r[1]]

    adsr = [
        np.linspace(0, adsr_a[0], adsr_t[0] * sample_rate, False),
        np.linspace(adsr_a[0], adsr_a[1], adsr_t[1] * sample_rate, False),
        np.linspace(adsr_a[1], adsr_a[2], adsr_t[2] * sample_rate, False),
        np.linspace(adsr_a[2], adsr_a[3], adsr_t[3] * sample_rate, False)
    ]

    return np.hstack(adsr)

def getNotePlayable(note, hrm_amp, note_period):
    steps = step_table[note[0]]

    if len(note) > 1:
        if note[1] == '#':
            steps += 1
        elif note[1] == '_':
            steps -= 1
        else:
            print(">> Error: Invalid note suffix! Only '#' and '_' are valid suffixes.")
            return 0

    note_freq = 440 * 2**(steps / 12)
    print('note f:', note_freq)

    t = np.linspace(0, note_period, note_period * sample_rate, False)
    note_values = np.zeros(t.shape)

    for i in range(7):
        note_values += np.sin(2 * np.pi * note_freq*(0.5*(1+i)) * t) * hrm_amp[i]

    note_playable = 2**15 / np.max(abs(note_values)) * (note_values * getADSR(sample_rate, note_period))

    return note_playable

def generateNoteStream(note_data, finger_num):
    note_holders = []
    note_stream = []

    holder_model = NoteHolder('model')
    holder_model.insertNote('c', 0)
    note_model = holder_model.note
    stream_slice = np.zeros(len(note_model)//16)


    for i in range(finger_num):
        note_holders.append(NoteHolder('finger'+str(i)))

    for notes in note_data:
        for beat in range(16):
            for note in notes:
                if note[0][-1] == '#' or note[0][-1] == '_':
                    note_arg = note[0][-2:]
                    note_beat = note[0][:-2]
                else:
                    note_arg = note[0][-1]
                    note_beat = note[0][:-1]

                note_duration = note[1]

                if int(note_beat) == beat:
                    for holder in note_holders:
                        if holder.isEmpty:
                            holder.insertNote(note_arg, note_duration)
                            break

            stream_slice = np.zeros(stream_slice.shape)

            for holder in note_holders:
                if not holder.isEmpty:
                    stream_slice += holder.popNoteSlice()

            note_stream.extend(stream_slice)

    return np.array(note_stream) * (2**15 * 0.8) / abs(np.max(note_stream))

def playStream(note_stream):
    playback = sa.play_buffer(note_stream, 1, 2, sample_rate)
    plt.plot(note_stream)
    plt.show()

    playback.wait_done()

def main():
    note_stream = generateNoteStream(sample_notes, finger_num)

    note_stream = note_stream.astype(np.int16)

    print(np.max(note_stream))
    print(len(note_stream), note_stream)
    playStream(note_stream)

if __name__ == '__main__':
    main()

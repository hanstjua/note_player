#!/usr/bin/python

import numpy as np
import simpleaudio as sa
import threading as th
import time

class SoundGenerator(th.Thread):
    def __init__(self, threadID):
        self.id = threadID

    def run(self):
        # play sound
        pass

class NoteHolder():
    def __init__(self, name, note, duration):
        self.name = name
        self.note = getNotePlayable(note, hrm_amp, note_period)
        self.note_len = len(self.note)
        self.timeleft = duration

    def popNoteSlice(self):
        if self.timeleft == 0:
            print(self.name, 'is empty!')
            return

        maxIdx = note_len//16
        note_slice = self.note[:maxIdx]
        self.note = self.note[maxIdx:]
        self.timeleft -= 1

        if(self.timeleft==0): self.reset()

        return note_slice

    def reset(self):
        self.note = np.zeros(self.note.shape)
        self.note_len = len(self.note)

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
    if ord(note) < ord('c'):
        steps = ord('c') - ord(note)
    else:
        steps = ord(note) - ord('c')

    note_freq = 440 * 2**0.25 * 2**(steps / 12)

    t = np.linspace(0, T, T * sample_rate, False)
    note_values = np.zeros(t.shape)

    for i in range(7):
        note_values += np.sin(2 * np.pi * note_freq*(0.5*(1+i)) * t) * hrm_amp[i]

    note_playable = 2**15 / np.max(abs(note_values)) * (note_values * getADSR(sample_rate, note_period))

    return note_playable

def generateNoteStream(note_data):
    while(len(note_data)):
        cur_notes = note_data[0]
        note_data = note_data[1:]

        # cur_notes =

def playStream(note_stream):
    playback = sa.play_buffer(note_stream, 1, 2, sample_rate)
    playback.wait_done()

def main():
    sample_notes = [
        [('c',4),('d',4),('e',4)],
        [('f',4)],
        [('g',4)]
    ]

    sample_rate = 44100
    note_period = 1.25

    hrm_amp = [0.17, 0.985, 0.185, 0.110, 0.26, 0.095, 0.080]  # timbre

    note_stream = generateNoteStream(sample_notes)

    playStream(note_stream)

if __name__ == '__main__':
    main()

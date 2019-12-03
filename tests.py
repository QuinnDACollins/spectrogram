# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:12:09 2019

@author: Quinn Collins
"""
from scipy.io.wavfile import read
from spectrogram import Spectrogram

musFs, musSignal = read("imissyou.wav")
mus_window_size = int(musFs * .02)
musicSpec = Spectrogram(mus_window_size, musSignal, 1.0)


speechFs, speechSignal = read("footmass.wav")
speech_window_size = int(speechFs * .02)
speechSpec = Spectrogram(speech_window_size, speechSignal, 1.0)

musicSpec.plot_spectrogram()
speechSpec.plot_spectrogram()


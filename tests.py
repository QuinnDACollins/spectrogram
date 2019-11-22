# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:12:09 2019

@author: Quinn Collins
"""
from scipy.io.wavfile import read
from spectrogram import Spectrogram

fs, signal = read("imissyou.wav")
window_size = int(fs * .02)
spec = Spectrogram(window_size, signal, 0.1)

spec.plot_spectrogram()
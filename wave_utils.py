# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:39:53 2019

@author: Quinn Collins
"""


from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import ft

   
#author: Quinn Collins
#generating wave amplitudes from complex number matrix
#[Inputs]   signal: a 1D time-domain numpy array containing the signal to perform dft on
            # and further calculate the amplitudes of the dft
#[Outputs]  y: a 1D numpy array containing the various magnitudes of the signals 
def gen_amplitudes(signal, twiddle_mat):
    fft_signal = ft.fft(signal, twiddle_mat)
    return np.abs(fft_signal)

    
        
    
    
        
        
        


    



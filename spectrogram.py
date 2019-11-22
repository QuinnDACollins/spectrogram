import matplotlib.pyplot as plt
import numpy as np
import wave_utils as wu

class Spectrogram:
    #Author: Quinn Collins
    #Inputs:
    #window_size: Size of window (How many frames are being analyzed at once)
    #signal: An audio signal (In our case generally from a .wav file)
    #overlap_ratio: The amount of the sliding window that you want to move OUT of your previous window
                    #This is a value from 0 - 1
    #Properties:
    #twiddle_mat: A matrix of your signal using for FFT
    #spectrogram: A matrix of zeros to be filled with your possible amplitudes per window slide
    
    
    def __init__(self, window_size, signal, overlap_ratio):
        ##INPUTS
        self.signal = signal
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        
        ##MATRICES
        self.twiddle_mat = np.full((window_size, window_size), 0j)
        for a in range(window_size):
            for b in range(window_size):
                self.twiddle_mat[a][b] += np.e**(-2j*(np.pi)*(a)*(b)/window_size)
        self.spectrogram = np.zeros((window_size // 2, (len(signal) // int(window_size * overlap_ratio))))
        
        
        
        
    #Author: Quinn Collins
    #Plots a spectrogram based on the values passed into the Spectrogram object
    def plot_spectrogram(self):
        #calculate the hop value, which decides how large the jump is from one window slide to another
        hop = int(self.overlap_ratio * self.window_size)
        
        #Create a hamming window for the purpose of normalizing some of the signals 
        w = np.hamming(self.window_size)
        
        #Outer loop: Loop through the entire signal based on our window size and our hop
                        #Calculate the amplitudes of the frames within that signal
        #Inner Loop: Loop through all of the y values (Initially zeroes) at our current 
                    # i value in our spectrogram and replace with the amplitudes calculated in outer
        for i in range(0, len(self.signal) - self.window_size, hop):
            amps = wu.gen_amplitudes(self.signal[i : i+ self.window_size]*w, self.twiddle_mat)
            idx = i // hop
            for y in range(self.window_size // 2):
                self.spectrogram[y][idx] = amps[y]
            
        #Plot the spectrogram
        plt.figure(figsize=(10, 10))
        plt.imshow(self.spectrogram[::-1]**.2)


# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:57:27 2019

@author: Quinn Collins
"""
import sys
sys.path.append('../')
from scipy.io import wavfile
import torch
from spectrogram import Spectrogram

def load_wav(file_path):
    fs, signal = wavfile.read(file_path)
    print(signal)
    #Check for multiple channels in the signal
    if len(signal.shape) == 1:
        return fs, signal
    elif len(signal.shape) == 2:
        return fs, signal[:,0]
    else:
        raise("Not a wav file!")

def load_data_from_wav(file_path, window_size, overlap_rat):
    fs, signal = load_wav(file_path)
    dataSpec = Spectrogram(window_size, signal, overlap_rat)
    return dataSpec.spectrogram


def load_data_from_wav_list(file_paths, window_size, overlap_rat):
    data_dim = [0, 0]
    temp_data = []
    for file_path in file_paths:
        #Load the spectrogram using our old function
        spectro = load_data_from_wav(file_path, window_size, overlap_rat)
        print(spectro)
        #Append our spectrogram and the # of windows from it to our temporary data
        temp_data.append((spectro, spectro.shape[1]))
        data_dim[0] = spectro.shape[0]
        #This dimension increases with every window from every spectrogram we test!
        data_dim[1] += spectro.shape[1]
    #Put all the data into vectors
    data = torch.zeros(data_dim)
    last = 0
    for dat, length in temp_data:
        #Ask justin about this.
        data[:,last:last + length] = torch.as_tensor(dat).float()
        last += length
    return data

def normalize(x, min_max = None):
    if min_max:
        return (x - min_max[0]) / (min_max[1] - min_max[0])
    else:
        return (x - x.min()) / (x.max() - x.min())
    
def find_min_max(x):
    return (min([y.min() for y in x]), max([y.max() for y in x]))

def exp_scale(x, y):
    return x**y
    
    
        
        
    
    

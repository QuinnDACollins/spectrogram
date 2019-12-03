# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:18:52 2019

@author: Justin Wang => Adapted to Quinn Collins spectrogram project
"""
import sys
sys.path.append('models/')
import nearest_centroid

sys.path.append('../')
import math
import data
import torch


######PRE PROCESSING###########

#FILE PATHS FOR OUR DATA#
voice_data_files = ['../royale_with_cheese.wav', '../footmass.wav']
music_data_files = ['../imissyou.wav']

###Parameters for our spectrogram###
window_size = 800
overlap_rat = 0.1

voice_data = data.load_data_from_wav_list(voice_data_files, window_size, overlap_rat)
music_data = data.load_data_from_wav_list(music_data_files, window_size, overlap_rat)

###Class Balancing (Making sure both our Voice Data and Music Data have the same amoutn of data for usage in our model)

#pick the one with the lower # of samples
num_samples_per_class = min(voice_data.shape[1], music_data.shape[1])
voice_data = voice_data[:,:num_samples_per_class]
music_data = music_data[:,:num_samples_per_class]

#Make data points into rows [this is a standard]
voice_data = torch.as_tensor(voice_data).t().float()
music_data = torch.as_tensor(music_data).t().float()

#Exponentially scale the data, (We did this during the meeting, made the spectrogram a lot more clear)
voice_data = data.exp_scale(voice_data, 0.2)
music_data = data.exp_scale(music_data, 0.2)

#Normalize
min_max = data.find_min_max([voice_data, music_data])
voice_data = data.normalize(voice_data, min_max=min_max)
music_data = data.normalize(music_data, min_max=min_max)

#split the data into training and validation(90% 10% split)
#Training: The data we feed the model 
#Validation: The data we use to see if our model predicts correctly

train_val_split = 0.9
train_num = math.floor(num_samples_per_class * train_val_split)

xtrain = torch.cat((voice_data[:train_num], music_data[:train_num]))
ytrain = torch.cat((torch.zeros(train_num), torch.ones(train_num)))

xval = torch.cat((voice_data[train_num:], music_data[train_num:]))
yval = torch.cat((torch.zeros(len(xval) // 2), torch.ones(len(xval) // 2)))

# Centroid Based Model #

#Model declaration
centroid_model = nearest_centroid.Model(xtrain.shape[1], 2)

#Train model
centroid_model.train(xtrain, ytrain)

#Test model
print("Accuracy: %.2f" % (100 * centroid_model.validate(xval, yval)))









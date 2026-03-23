#Author: Kyle Frohlich
#Name: CNN of CCR5 Delta-32
#Date: 06Mar2026
#Purpose: Architecture of the model

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

class CCR5CNN(nn.Module):

    def __init__(self):

        super().__init__()

        #inChannels is 4 for the 4 nucleotides A, C, G, and T
        #outChannels is 16 because the model is fairly small
        #kernelSize set at 7 as part of preparation and testing of the model. This is to
        #allow for sensing motifs without overcrowding the filters.
        self.convolution1 = nn.Conv1d(in_channels = 4, out_channels = 16, kernel_size = 8)
        #pool set at 1 as we are looking for just the highest value filter activated 
        self.pool = nn.AdaptiveMaxPool1d(1)
        #uses our 16 outchannels to get one value out
        self.fcLayer = nn.Linear(16,1)


    def forward(self, x):
        
        x = func.relu(self.convolution1(x))
        #takes the strongest activation for each filter
        x = self.pool(x)
        #removes dimension from pooling
        x = x.squeeze(-1)
        #combines all 16 signals from convolution
        x = self.fcLayer(x)

        return x.squeeze(1)
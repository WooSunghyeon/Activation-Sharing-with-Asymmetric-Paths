import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from building_blocks import *

class Dfa_Block(nn.Module): # for DFA in ResNet
    def __init__(self, connect_channels):
        super(Dfa_Block, self).__init__()
        self.dfa = Feedback_Receiver(connect_channels)
    
    def forward(self, x):
        x, dm1 = self.dfa(x)
        self.dummy = dm1
        return x
    
class ASAP_Conv_Block(nn.Module): # for ASAP in ConvNet
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, wt = False):
        super(ASAP_Conv_Block, self).__init__()
        
        self.conv = Conv2d_FA_ASAP(in_channels, out_channels, kernel_size, stride, padding, wt = wt)
        self.bn = nn.BatchNorm2d(out_channels)
        self.max = nn.MaxPool2d(2,2, ceil_mode = True)
        
    def forward(self, x, save):
        x = self.conv(x, save)
        x = self.max(F.relu(self.bn(x)))
        return x
        

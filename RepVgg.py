#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 20:16:21 2022

@author: ding
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torchsummary

class repvgg_setting(object):
    def __init__(self, model_type):
        ab_set = {'A0': [.75, 2.5], 'A1': [1, 2.5], 'A3': [1.5, 2.75],
                  'B0': [1, 2.5], 'B1': [2, 4], 'B2':[2.5, 5], 'B3':[3, 5]}
        if model_type not in ab_set:
            raise ValueError('type error')
        a, b = ab_set.get(model_type)
        self.layers = np.array([1, 2, 4, 14, 1] if model_type[0] == 'A' else [1, 4, 6, 16, 1])
        layer_param = [1, 64, 128, 256, 512]
        a0 = min(64, 64*a)
        self.channels = (np.concatenate(([a0], np.repeat(a, 3), [b])) * np.array(layer_param)).astype('int')
        
class RepVgg_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_deploy=False):
        super(RepVgg_Block, self).__init__()
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.residual = nn.BatchNorm2d(in_channels) if stride == 1 else None
        self.activate = nn.ReLU()
        self.stride = stride
        
    def forward(self, x):
        out = self.conv_3x3(x) + self.conv_1x1(x)
        if self.stride == 1:
            out += self.residual(x)
        out = self.activate(out)
        return(out)


class MyRepVgg(nn.Module):
    def __init__(self, stages, channels, num_classes=1000):
        super(MyRepVgg, self).__init__()
        
        self.in_channels = channels[0]
        
        self.stage0 = RepVgg_Block(in_channels=3, out_channels=channels[0], stride=2)
        self.stage1 = self.make_stage(num_blocks = stages[1], out_channels=channels[1])
        self.stage2 = self.make_stage(num_blocks = stages[2], out_channels=channels[2])
        self.stage3 = self.make_stage(num_blocks = stages[3], out_channels=channels[3])
        self.stage4 = self.make_stage(num_blocks = stages[4], out_channels=channels[4])
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def make_stage(self, num_blocks, out_channels):
        blocks = [RepVgg_Block(self.in_channels, out_channels, stride=2)]
        for i in range(num_blocks-1):
            blocks.append(RepVgg_Block(out_channels, out_channels, stride=1))
        self.in_channels = out_channels
        
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
if __name__ == '__main__':

    layers, channels = repvgg_setting('A0').layers, repvgg_setting('A0').channels
    model = MyRepVgg(layers, channels)
    torchsummary.summary(model, input_size=(3,224,224), batch_size=1, device='cpu')
# -*- coding: utf-8 -*-

# @Description: 
# @Author: CaptainHu
# @Date: 2021-07-06 17:46:51
# @LastEditors: CaptainHu

import torch
import torch.nn as nn
import torch.nn.functional as F

import debug_tools as D

class FCDDClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # CUT
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.features=self.features[:-8]

        self.cls_head=nn.Sequential(
            nn.Conv2d(512,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(256,64,1,1,0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,1,1,1,0),
        )

    def forward(self, x):
        x_f=self.features(x)
        try:
            y=self.cls_head(x_f)
        except:
            breakpoint()
        y=torch.sigmoid(y)
        return y,x_f



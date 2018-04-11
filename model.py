import torch
import torch.nn as nn
import math
import pdb
from collections import OrderedDict



class QuadNet(nn.Module):
    def __init__(self, num_classes=10):
        super(QuadNet, self).__init__()

        self.featReal = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(100, 150, kernel_size=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(150, 250, kernel_size=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.fcReal = nn.Sequential(
            nn.Linear(3*3*250, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, num_classes),
        )

        self.featTemp = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(100, 150, kernel_size=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(150, 250, kernel_size=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.fcTemp = nn.Sequential(
            nn.Linear(3*3*250, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_real(self, x):
        x = self.featReal(x)
        x = x.view(-1, 3*3*250)
        x = self.fcReal(x)
        return x

    def forward_temp(self, x):
        x = self.featTemp(x)
        x = x.view(-1, 3*3*250)
        x = self.fcTemp(x)
        return x

    def forward(self, realA, realB, tempA, tempB):        
        RA = self.forward_real(realA)
        RB = self.forward_real(realB)
        TA = self.forward_temp(tempA)
        TB = self.forward_temp(tempB)
        return RA, RB, TA, TB


class QuadNetSingle(nn.Module):
    def __init__(self, num_classes=10):
        super(QuadNetSingle, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, kernel_size=7, padding=0)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=4, padding=0)

        self.fc1 = nn.Linear(3*3*250, 300)
        self.fc2 = nn.Linear(300, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_once(self, x):
        x = self.conv1(x)   # 100x42x42
        x = self.relu(x)    # 100x42x42
        x1 = self.pool(x)   # 100x21x21

        x = self.conv2(x1)  # 150x18x18
        x = self.relu(x)    # 150x18x18
        x2 = self.pool(x)   # 150x9x9

        x = self.conv3(x2)  # 250x6x6
        x = self.relu(x)    # 250x6x6
        x3 = self.pool(x)   # 250x3x3

        xv = x3.view(-1, 3*3*250)
        xfc1 = self.relu(self.fc1(xv))
        output = self.fc2(xfc1)
        
        return output

    def forward(self, realA, realB, tempA, tempB):        
        RA = self.forward_once(realA)
        RB = self.forward_once(realB)
        TA = self.forward_once(tempA)
        TB = self.forward_once(tempB)
        return RA, RB, TA, TB
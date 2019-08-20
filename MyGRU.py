import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class diceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smoth = 1e-5

    def forward(self, output, target):
        target0 = torch.ones(target.shape).cuda()
        target0[target==1] = 0
        target1 = torch.ones(target.shape).cuda()
        target1[target==0] = 0

        intersection0 = 2. * torch.sum(output[:,0,:,:,:] * target0)
        denominator0 = torch.sum(output[:,0,:,:,:] * output[:,0,:,:,:]) + torch.sum(target0 * target0)
        dice0 = (intersection0 + self.smoth) / (denominator0 + self.smoth)
        intersection1 = 2. * torch.sum(output[:, 1, :, :, :] * target1)
        denominator1 = torch.sum(output[:, 1, :, :, :] * output[:, 1, :, :, :]) + torch.sum(target1 * target1)
        dice1 = (intersection1 + self.smoth) / (denominator1 + self.smoth)

        dice = 1 - 0.5*(dice0+dice1)

        return dice
class GRUCell(nn.Module):
    def __init__(self,channel=2):
        super(GRUCell, self).__init__()
        # the channel of in and out must be equal and equal to the class
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=3,stride=1,padding=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        reset_1 = self.tanh(self.conv3d(input)+self.conv3d(hidden))
        reset_2 = self.tanh(self.conv3d(input)+self.conv3d(hidden))

        u_1 = torch.mul(reset_1,self.conv3d(input))+torch.mul((1-reset_1),input)
        u_2 = torch.mul(reset_2,self.conv3d(u_1))+torch.mul((1-reset_1),u_1)
        u_3 = self.conv3d(u_2)
        u_3 = self.softmax(u_3)
        return u_3

class GRUModel(nn.Module):
    def __init__(self,num_layers,channel=2):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.GRUCell = GRUCell(channel)

    def forward(self, input, hidden):
        # filters = torch.ones(2, 1, 1, 1, 1).cuda()
        # input = F.conv3d(input, filters)
        # hidden = F.conv3d(hidden, filters)
        for _ in range(self.num_layers):
            hidden = self.GRUCell(input, hidden)
        return hidden
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, output_len=100,in_channels=1, out_channels=32, device=None, **kwargs):
        super().__init__()

        nc = 80
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c3 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        #self.d4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c41 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c42 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c43 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)

        self.c5 = nn.Conv2d(in_channels=nc, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        
        #self.conv_net_global = nn.Sequential(
        #    #2048 -> 1024
        #    #nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
        #    #nn.ReLU(),
        #    #1024 -> 512
        #    nn.GELU(),
        #    #-> 256
        #    nn.GELU(),
        #    #-> 128
        #    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        #    nn.GELU(),
        #    #-> 64
        #    #nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        #    nn.Conv1d(in_channels=256, out_channels=sum(self.channel_splits), kernel_size=3, stride=2, padding=1),
        #)

    def forward(self, x):

        x = self.c1(x)
        _x = x
        x = F.relu(x) 

        x = self.c2(x) + _x 
        _x = x
        x = F.relu(x) 

        x = self.c3(x) + _x
        _x = x
        x = F.relu(x)


        x = self.c4(x) + _x
        _x = x
        x = F.relu(x)

        x = self.c41(x) + _x
        _x = x
        x = F.relu(x)

        x = self.c42(x) + _x
        _x = x
        x = F.relu(x)

        #_x = x
        #x = self.c4(x)
        #x = F.relu(x)+ self.d4(_x)

        x = self.c5(x)

        return x
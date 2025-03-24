import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, output_len=100,in_channels=1, out_channels=32, device=None, **kwargs):
        super().__init__()

        nc = 128
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c3 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        #self.d4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c41 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c42 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c43 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c44 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c45 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c46 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c47 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)
        self.c48 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2)

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

        x = self.c43(x) + _x
        _x = x
        x = F.relu(x)

        x = self.c44(x) + _x
        _x = x
        x = F.relu(x)

        x = self.c45(x) + _x
        _x = x
        x = F.relu(x)

        x = self.c46(x) + _x
        _x = x
        x = F.relu(x)

        #_x = x
        #x = self.c4(x)
        #x = F.relu(x)+ self.d4(_x)

        x = self.c5(x)

        return x

class Resnet1dBlock(nn.Module):
    def __init__(self, in_channels, activation=True):
        super().__init__()
        pm = 'zeros'
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2, padding_mode=pm)
        self.shortcut = nn.Conv1d(in_channels, in_channels, 1)
        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.activation = activation
        self.in_channels = in_channels


    def forward(self, x):
        batchsize = x.shape[0]
        size_x = x.shape[2]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.in_channels, -1)).view(batchsize, self.in_channels, size_x)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, output_len=100,in_channels=1, out_channels=32, device=None, **kwargs):
        super().__init__()

        n_layers = 10
        width = 100
        layers = [Resnet1dBlock(width) for i in range(n_layers - 1)]
        self.net = nn.Sequential(*layers)

        self.in_conv = nn.Conv1d(in_channels, width, kernel_size=5, stride=1, padding=2)
        self.out_conv = nn.Conv1d(width, out_channels, kernel_size=5, stride=1, padding=2)

        self.fc0 = nn.Linear(in_channels, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.fc0(x)
        x = x.permute(0,3,1,2)

        #x = self.in_conv(x)
        #x = torch.relu(x)
        x = self.net(x)
        #x = self.out_conv(x)


        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)

        return x

class Resnet2dBlock(nn.Module):
    def __init__(self, in_channels, activation=True):
        super().__init__()
        pm = 'zeros'
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2, padding_mode=pm)
        self.shortcut = nn.Conv1d(in_channels, in_channels, 1)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.activation = activation
        self.in_channels = in_channels


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.in_channels, -1)).view(batchsize, self.in_channels, size_x, size_y)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out

class ResNet2D(nn.Module):
    def __init__(self, output_len=100,in_channels=1, out_channels=32, device=None, **kwargs):
        super().__init__()

        n_layers = 10
        width = 100
        layers = [Resnet2dBlock(width) for i in range(n_layers - 1)]
        self.net = nn.Sequential(*layers)

        self.in_conv = nn.Conv2d(in_channels, width, kernel_size=5, stride=1, padding=2)
        self.out_conv = nn.Conv2d(width, out_channels, kernel_size=5, stride=1, padding=2)

        self.fc0 = nn.Linear(in_channels, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.fc0(x)
        x = x.permute(0,3,1,2)

        #x = self.in_conv(x)
        #x = torch.relu(x)
        x = self.net(x)
        #x = self.out_conv(x)


        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)

        return x

class Resnet3dBlock(nn.Module):
    def __init__(self, in_channels, activation=True):
        super().__init__()
        pm = 'zeros'
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2)
        self.shortcut = nn.Conv1d(in_channels, in_channels, 1)
        self.bn = torch.nn.BatchNorm3d(in_channels)
        self.activation = activation
        self.in_channels = in_channels

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[2], x.shape[3], x.shape[4]

        out = self.conv(x)
        out += self.shortcut(x.view(batchsize, self.in_channels, -1)).view(batchsize, self.in_channels, size_x, size_y, size_z)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, output_len=100,in_channels=1, out_channels=32, device=None, **kwargs):
        super().__init__()

        n_layers = 8
        width = 64
        pm = 'zeros'
        self.in_conv = nn.Conv3d(in_channels, width, kernel_size=5, stride=1, padding=2)
        self.out_conv = nn.Conv3d(width, out_channels, kernel_size=5, stride=1, padding=2)
        layers = [Resnet3dBlock(width) for i in range(n_layers - 1)]
        self.net = nn.Sequential(*layers)

        self.fc0 = nn.Linear(in_channels, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        x = self.fc0(x)
        x = x.permute(0,4,1,2,3)
        #x = self.in_conv(x)
        #x = torch.relu(x)

        x = self.net(x)
        #x = self.out_conv(x)

        x = x.permute(0,2,3,4,1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0,4,1,2,3)

        return x

#class ResNet3D(nn.Module):
#    def __init__(self, output_len=100,in_channels=1, out_channels=32, device=None, **kwargs):
#        super().__init__()
#
#        nc = 64
#        pm = 'circular'
#        self.c1 = nn.Conv3d(in_channels=in_channels, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c2 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c3 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        #self.d4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
#        self.c4 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c41 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c42 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c43 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c44 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c45 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c46 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c47 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#        self.c48 = nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=5, stride=1, padding=2, padding_mode=pm)
#
#        self.c5 = nn.Conv3d(in_channels=nc, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
#        
#        #self.conv_net_global = nn.Sequential(
#        #    #2048 -> 1024
#        #    #nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
#        #    #nn.ReLU(),
#        #    #1024 -> 512
#        #    nn.GELU(),
#        #    #-> 256
#        #    nn.GELU(),
#        #    #-> 128
#        #    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
#        #    nn.GELU(),
#        #    #-> 64
#        #    #nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
#        #    nn.Conv1d(in_channels=256, out_channels=sum(self.channel_splits), kernel_size=3, stride=2, padding=1),
#        #)
#
#    def forward(self, x):
#
#        x = self.c1(x)
#        _x = x
#        x = F.elu(x) 
#
#        x = self.c2(x) + _x 
#        _x = x
#        x = F.elu(x) 
#
#        x = self.c3(x) + _x
#        _x = x
#        x = F.elu(x)
#
#
#        x = self.c4(x) + _x
#        _x = x
#        x = F.elu(x)
#
#        x = self.c41(x) + _x
#        _x = x
#        x = F.elu(x)
#
#        x = self.c42(x) + _x
#        _x = x
#        x = F.elu(x)
#
#        #x = self.c43(x) + _x
#        #_x = x
#        #x = F.elu(x)
#
#        #x = self.c44(x) + _x
#        #_x = x
#        #x = F.elu(x)
#
#        x = self.c45(x) + _x
#        _x = x
#        x = F.elu(x)
#
#        x = self.c46(x) + _x
#        _x = x
#        x = F.elu(x)
#
#        #_x = x
#        #x = self.c4(x)
#        #x = F.relu(x)+ self.d4(_x)
#
#        x = self.c5(x)
#
#        return x
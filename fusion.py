import torch
import torch.nn as nn

class AVFFAttention(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AVFFAttention, self).__init__()
        inter_channels = int(channels // r)

        self.conv_in = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0)

        self.conv_cat = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0)

        # self.conv_plus = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, t):

        x = self.conv_in(x)

        xl = torch.cat([x, t], 1)
        xl = self.conv_cat(xl)
        xl = self.local_att(xl)

        x2 = x + t
        x2 = self.global_att(x2)

        x3 = xl + x2
        wei = self.sigmoid(x3)

        out = 2 * x * wei + 2 * t * (1 - wei)

        return out    

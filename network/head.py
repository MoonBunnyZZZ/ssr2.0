import torch
import torch.nn as nn


class StageRegression(nn.Module):
    def __init__(self, in1_chs, in2_chs, pool_size):
        super(StageRegression, self).__init__()
        self.fb1 = nn.Sequential(nn.Conv2d(in1_chs, 10, 1, 1),
                                 nn.LeakyReLU(inplace=True),
                                 nn.AvgPool2d(pool_size, ceil_mode=True),
                                 nn.Flatten())
        self.pb1 = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(160, 10), nn.LeakyReLU(inplace=True))
        self.fb2 = nn.Sequential(nn.Conv2d(in2_chs, 10, 1, 1),
                                 nn.LeakyReLU(inplace=True),
                                 nn.MaxPool2d(pool_size, ceil_mode=True),
                                 nn.Flatten())
        self.pb2 = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(160, 10), nn.LeakyReLU(inplace=True))
        self.delta = nn.Sequential(nn.Linear(160, 1), nn.Tanh())
        self.feature = nn.Sequential(nn.Linear(10, 6), nn.Tanh())
        self.eta = nn.Sequential(nn.Linear(6, 3), nn.Tanh())
        self.p = nn.Sequential(nn.Linear(6, 3), nn.Tanh())

    def forward(self, x1, x2):
        fb1 = self.fb1(x1)
        pb1 = self.pb1(fb1)

        fb2 = self.fb2(x2)
        pb2 = self.pb2(fb2)

        delta = self.delta(torch.mul(fb1, fb2))
        feature = self.feature(torch.mul(pb1, pb2))
        eta = self.eta(feature)
        p = self.p(feature)

        return delta, eta, p


class FPN(nn.Module):
    def __init__(self, base_chs):
        super(FPN, self).__init__()
        self.up_sample_3to2 = nn.Sequential(nn.Conv2d(base_chs, base_chs // 2, 1),
                                            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.up_sample_2to1 = nn.Sequential(nn.Conv2d(base_chs, base_chs // 4, 1),
                                            nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        x1, x2, x3 = x

        upsampled_x3 = self.up_sample_3to2(x3)
        new_x2 = torch.cat((x2, upsampled_x3), 1)

        upsampled_x2 = self.up_sample_2to1(new_x2)
        new_x1 = torch.cat((x1, upsampled_x2), 1)

        return [new_x1, new_x2, x3]

import torch
import torch.nn as nn

from network.backbone import ResNet
from network.head import StageRegression

stream1_cfg = dict(stem=dict(out_channels=32, kernel_size=7, stride=2, padding=0),
                   stage1=dict(in_chs=32, out_chs=32, stride=1, depth=2),
                   stage2=dict(in_chs=32, out_chs=64, stride=2, depth=2),
                   stage3=dict(in_chs=64, out_chs=128, stride=2, depth=2),
                   stage4=dict(in_chs=128, out_chs=256, stride=2, depth=2))

stream2_cfg = dict(stem=dict(out_channels=16, kernel_size=7, stride=2, padding=0),
                   stage1=dict(in_chs=16, out_chs=16, stride=1, depth=2),
                   stage2=dict(in_chs=16, out_chs=32, stride=2, depth=2),
                   stage3=dict(in_chs=32, out_chs=64, stride=2, depth=2),
                   stage4=dict(in_chs=64, out_chs=128, stride=2, depth=2))


class SSR(nn.Module):
    def __init__(self):
        super(SSR, self).__init__()
        self.stream1 = ResNet(stream1_cfg)
        self.stream2 = ResNet(stream2_cfg)
        self.stage1 = StageRegression(64, 32, 8)
        self.stage2 = StageRegression(128, 64, 4)
        self.stage3 = StageRegression(256, 128, 2)

    def forward(self, x):
        y1 = self.stream1(x)
        y2 = self.stream2(x)

        delta1, eta1, p1 = self.stage1(y1[0], y2[0])
        delta2, eta2, p2 = self.stage2(y1[1], y2[1])
        delta3, eta3, p3 = self.stage3(y1[2], y2[2])
        # print(delta1.size(), eta1.size(), p1.size())
        # print(delta2.size(), eta2.size(), p2.size())
        # print(delta3.size(), eta3.size(), p3.size())

        s1 = (3 * (1 + delta1))
        s2 = (3 * (1 + delta2))
        s3 = (3 * (1 + delta3))
        # print(s1, s2, s3)

        a = torch.sum(torch.mul(input=torch.add(input=torch.Tensor([[0, 1, 2]]).cuda(),
                                                other=eta3),
                                other=p3), axis=1, keepdims=True)
        b = torch.sum(torch.mul(input=torch.add(input=torch.Tensor([[0, 1, 2]]).cuda(),
                                                other=eta2),
                                other=p2), axis=1, keepdims=True)
        c = torch.sum(torch.mul(input=torch.add(input=torch.Tensor([[0, 1, 2]]).cuda(),
                                                other=eta1),
                                other=p1), axis=1, keepdims=True)
        # print(a, '\n', b, '\n', c)
        age = 101 * (a / s3 + b / s2 / s3 + c / s1 / s2 / s3)
        # print('age', age)
        return age


if __name__ == '__main__':
    net = SSR()
    dummy = torch.rand((2, 3, 108, 108))
    with torch.no_grad():
        output = net(dummy)

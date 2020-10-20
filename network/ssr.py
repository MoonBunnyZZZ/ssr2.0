import torch
import torch.nn as nn

from network.backbone import ResNet
from network.head import StageRegression, FPN


class SSR(nn.Module):
    def __init__(self, stream1_cfg, stream2_cfg, fpn=False):
        super(SSR, self).__init__()
        self.stream1 = ResNet(stream1_cfg['backbone'])
        self.stream2 = ResNet(stream2_cfg['backbone'])
        self.fpn = fpn
        if fpn:
            self.fpn1 = FPN(stream1_cfg['head']['head3'])
            self.fpn2 = FPN(stream2_cfg['head']['head3'])
            self.stage1 = StageRegression(stream1_cfg['head']['head1']*2, stream2_cfg['head']['head1']*2, 8)
            self.stage2 = StageRegression(stream1_cfg['head']['head2']*2, stream2_cfg['head']['head2']*2, 4)
        else:
            self.stage1 = StageRegression(stream1_cfg['head']['head1'], stream2_cfg['head']['head1'], 8)
            self.stage2 = StageRegression(stream1_cfg['head']['head2'], stream2_cfg['head']['head2'], 4)
        self.stage3 = StageRegression(stream1_cfg['head']['head3'], stream2_cfg['head']['head3'], 2)

    def forward(self, x):
        y1 = self.stream1(x)
        y2 = self.stream2(x)
        if self.fpn:
            y1 = self.fpn1(y1)
            y2 = self.fpn2(y2)
        delta1, eta1, p1 = self.stage1(y1[0], y2[0])
        delta2, eta2, p2 = self.stage2(y1[1], y2[1])
        delta3, eta3, p3 = self.stage3(y1[2], y2[2])

        s1 = (3 * (1 + delta1))
        s2 = (3 * (1 + delta2))
        s3 = (3 * (1 + delta3))

        a = torch.sum(torch.mul(input=torch.add(input=torch.Tensor([[0, 1, 2]]).cuda(),
                                                other=eta3),
                                other=p3), axis=1, keepdims=True)
        b = torch.sum(torch.mul(input=torch.add(input=torch.Tensor([[0, 1, 2]]).cuda(),
                                                other=eta2),
                                other=p2), axis=1, keepdims=True)
        c = torch.sum(torch.mul(input=torch.add(input=torch.Tensor([[0, 1, 2]]).cuda(),
                                                other=eta1),
                                other=p1), axis=1, keepdims=True)
        age = 101 * (a / s3 + b / s2 / s3 + c / s1 / s2 / s3)
        return age



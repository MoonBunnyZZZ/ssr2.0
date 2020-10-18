import torch
import torch.nn as nn
import torch.distributed as dist


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mae(output, target):
    output = output.clone()
    return torch.sum(torch.abs(output - target))


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def init_model(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.period_sum = 0
        self.period_count = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.period_sum += val * n
        self.period_count += n

    def period_avg(self):
        avg = self.period_sum / self.period_count
        self.period_sum = 0
        self.period_count = 0
        return avg

    def avg(self):
        avg = self.sum / self.count
        return avg


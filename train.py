import os
import math
import argparse
import time
import torch
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from network.ssr import SSR
from dataset.dali_pipe import TrainPipe, ValPipe
from utils import AverageMeter, to_python_float

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads-num', type=int)
    parser.add_argument('--gpu-id', type=int)
    parser.add_argument('--gpus-num', type=int)
    parser.add_argument('--db-dir', type=str)
    parser.add_argument('--print-freq', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--warmup-num', type=int)
    parser.add_argument('--lr-decay-milestone', type=str)
    parser.add_argument('--max-epoch', type=int)
    parser.add_argument('--save-dir', type=str)

    arguments = parser.parse_args()
    return arguments


def lr_scheduler(args):
    warmup_num = args.warmup_num
    lr_decay_milestone = list(map(lambda x: int(x), args.lr_decay_milestone.split(',')))

    def scheduler_func(epoch):
        ratio = 1.0
        if epoch < warmup_num:
            return (epoch+1) / warmup_num
        else:
            for step in lr_decay_milestone:
                if epoch > step:
                    ratio *= 0.1
            return ratio

    return scheduler_func


def save_checkpoint(net_state, optimizer_state, save_dir, epoch, is_best=False):
    torch.save(net_state, '{}/net_epoch{}.pth'.format(save_dir, epoch))
    torch.save(optimizer_state, '{}/optimizer_epoch{}.pth'.format(save_dir, epoch))
    if is_best:
        torch.save({'net_state': net_state, 'epoch': epoch}, '{}/best.pth'.format(save_dir))


def get_dali_iterator(args):
    train_pipe = TrainPipe(args.batch_size, args.threads_num, args.gpu_id, args.gpus_num, args.db_dir)
    train_pipe.build()
    train_loader = DALIClassificationIterator([train_pipe], size=train_pipe.epoch_size("Reader"))

    val_pipe = ValPipe(args.batch_size, args.threads_num, args.gpu_id, args.gpus_num, args.db_dir)
    val_pipe.build()
    val_loader = DALIClassificationIterator([val_pipe], size=train_pipe.epoch_size("Reader"))

    return train_loader, val_loader


def get_net():
    net = SSR()
    net = net.cuda()
    return net


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, batch in enumerate(train_loader):
        data = batch[0]["data"]
        target = batch[0]["label"].contiguous().view(data.size(0), 1)
        target = target.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(data)
        loss = criterion(output, target)
        losses.update(to_python_float(loss), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        if i % args.print_freq == 0 and i > 1:
            print('{time} Epoch:[{0}] Step:[{1}] Time {2:.3f} '
                  'Data {3:.3f} Loss {4:.4f}'.format(epoch, i, batch_time.period_avg(), data_time.period_avg(),
                                                     losses.period_avg(), time=time.strftime("%m-%d %H:%M:%S",
                                                                                             time.localtime())))
    print('Epoch:[{0}] Train Loss {1:.4f} Now LR {2:6f}'.format(epoch, losses.avg(), optimizer.param_groups[0]["lr"]))
    return batch_time.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, batch in enumerate(val_loader):
        data = batch[0]["data"]
        target = batch[0]["label"].view(data.size(0), 1)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        losses.update(to_python_float(loss), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch [{}] Test Loss {:.4f}'.format(epoch, losses.avg()))

    return losses.avg()


def main():
    best_prec = 30.0
    args = cli()
    print(args)
    train_loader, val_loader = get_dali_iterator(args)
    net = get_net()
    criterion = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler(args))
    for epoch in range(1, args.max_epoch):
        train(train_loader, net, criterion, optimizer, epoch, args)
        prec = validate(val_loader, net, criterion, epoch)
        scheduler.step()
        if prec < best_prec or epoch % 5 == 0:
            is_best = prec < best_prec
            best_prec = min(prec, best_prec)
            save_checkpoint(net.state_dict(), optimizer.state_dict(), args.save_dir, epoch, is_best)

        # reset DALI iterators
        train_loader.reset()
        val_loader.reset()


if __name__ == '__main__':
    main()

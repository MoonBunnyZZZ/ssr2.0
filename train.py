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
    arguments = parser.parse_args()
    return arguments


def get_dali_iterator(args):
    train_pipe = TrainPipe(args.batch_size, args.threads_num, args.gpu_id, args.gpus_num, args.db_dir)
    train_pipe.build()
    train_loader = DALIClassificationIterator([train_pipe], size=train_pipe.epoch_size("Reader"))

    val_pipe = ValPipe(args.batch_size, args.threads_num, args.gpu_id, args.gpus_num, args.db_dir)
    val_pipe.build()
    val_loader = DALIClassificationIterator([val_pipe], size=train_pipe.epoch_size("Reader"))

    return train_loader, val_loader


def get_net(args):
    net = SSR()
    net = net.cuda()
    return net


def train(train_loader, model, criterion, optimizer, epoch, args):
    global global_step
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        # print(loss)

        losses.update(to_python_float(loss.clone().detach()), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        global_step += 1
        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('{time} Epoch: [{0}][{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i,
                                                                  batch_time=batch_time, data_time=data_time,
                                                                  loss=losses, time=time.strftime("%m-%d %H:%M:%S",
                                                                                                  time.localtime())))
    return batch_time.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.batch_size)

        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        losses.update(to_python_float(loss.clone()), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, val_loader_len,
                                                                  args.batch_size / batch_time.val,
                                                                  args.batch_size / batch_time.avg,
                                                                  batch_time=batch_time, loss=losses))

    return losses.avg


def main():
    args = cli()
    train_loader, val_loader = get_dali_iterator(args)
    net = get_net(args)
    criterion = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), args.lr)
    for epoch in range(0, 100):
        train(train_loader, net, criterion, optimizer, epoch, args)
        prec = validate(val_loader, model, criterion)

        if epoch % 5 == 0 or prec < best_prec:
            is_best = prec < best_prec
            best_prec = min(prec, best_prec)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save, epoch)

        # reset DALI iterators
        train_loader.reset()
        val_loader.reset()


if __name__ == '__main__':
    main()

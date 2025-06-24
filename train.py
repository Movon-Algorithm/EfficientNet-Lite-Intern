import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
from PIL import Image
import time
import logging
from efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite
from utils.train_utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

CROP_PADDING = 32
MEAN_RGB = [0.498, 0.498, 0.498]
STDDEV_RGB = [0.502, 0.502, 0.502]

class DataIterator(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def parse_device(device_str):
    device_str = device_str.lower()
    if device_str == 'cpu':
        return torch.device('cpu')
    elif device_str.isdigit():
        return torch.device(f'cuda:{device_str}')
    else:
        raise ValueError(f"Invalid device string: {device_str}. Use 'cpu' or GPU index as integer string like '0', '1', ...")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='efficientnet_lite0',
                        help='name of model: efficientnet_lite0, 1, 2, 3, 4')
    parser.add_argument('--device', type=str, default='0',
                        help="device to use, 'cpu' or GPU index like '0', '1', '2', ...")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_resume', type=str, default='./efficientnet_lite0.pth', help='path for eval model')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--total_iters', type=int, default=300000, help='total iters')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--auto_continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--display_interval', type=int, default=20, help='display interval')
    parser.add_argument('--val_interval', type=int, default=10000, help='val interval')
    parser.add_argument('--save_interval', type=int, default=10000, help='save interval')
    parser.add_argument('--train_dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='data/val', help='path to validation dataset')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    input_size = efficientnet_lite_params[args.model_name][2]

    # 원래 device 파싱
    device = parse_device(args.device)
    use_gpu = (device.type == 'cuda') and torch.cuda.is_available()

    # **여기서 핵심**: CUDA_VISIBLE_DEVICES 환경변수 있으면 내부 device 항상 cuda:0 으로 강제
    if use_gpu and 'CUDA_VISIBLE_DEVICES' in os.environ:
        device_id = 0
        device = torch.device(f'cuda:{device_id}')
    else:
        device_id = device.index if device.index is not None else 0

    print(f"Using device: {device}")

    assert os.path.exists(args.train_dir)
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_RGB, STDDEV_RGB)
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_gpu)
    train_dataprovider = DataIterator(train_loader)

    assert os.path.exists(args.val_dir)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(input_size + CROP_PADDING, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_RGB, STDDEV_RGB)
        ])),
        batch_size=200,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')

    model = build_efficientnet_lite(args.model_name, args.num_classes)

    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)

    if use_gpu:
        model = model.to(device)
        model = torch.nn.DataParallel(model)  # 여러 GPU 사용하려면 여기서 래핑
        loss_function = criterion_smooth.to(device)
    else:
        loss_function = criterion_smooth

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step: (1.0 - step / args.total_iters) if step <= args.total_iters else 0,
                    last_epoch=-1)

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            if use_gpu:
                # CUDA_VISIBLE_DEVICES=1 환경에서 내부 device는 cuda:0 이므로 map_location도 cuda:0 으로 지정
                checkpoint = torch.load(lastest_model, map_location='cuda:0')
            else:
                checkpoint = torch.load(lastest_model, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            load_checkpoint(model, checkpoint)
            validate(model, device, args, all_iters=all_iters)
        exit(0)

    while all_iters < args.total_iters:
        all_iters = train(model, device, args, val_interval=args.val_interval, bn_process=False, all_iters=all_iters)
        validate(model, device, args, all_iters=all_iters)
    all_iters = train(model, device, args, val_interval=int(1280000 / args.batch_size), bn_process=True, all_iters=all_iters)
    validate(model, device, args, all_iters=all_iters)
    save_checkpoint({'state_dict': model.state_dict(), }, args.total_iters, tag='bnps-')

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, device, args, *, val_interval, bn_process=False, all_iters=None):
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    for iters in range(1, val_interval + 1):
        scheduler.step()
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st

        output = model(data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        precisions = accuracy(output, target, topk=(1, 5))
        prec1 = precisions[0]
        prec5 = precisions[1] if len(precisions) > 1 else torch.tensor(0.0, device=prec1.device)

        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100

        if all_iters % args.display_interval == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
            logging.info(printInfo)
            t1 = time.time()
            Top1_err, Top5_err = 0.0, 0.0

        if all_iters % args.save_interval == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                }, all_iters)

    return all_iters

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 250
    t1  = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_function(output, target)

            precisions = accuracy(output, target, topk=(1, 5))
            prec1 = precisions[0]
            prec5 = precisions[1] if len(precisions) > 1 else torch.tensor(0.0, device=prec1.device)

            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)

def load_checkpoint(net, checkpoint):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.'+k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]

    net.load_state_dict(temp, strict=True)

if __name__ == "__main__":
    main()

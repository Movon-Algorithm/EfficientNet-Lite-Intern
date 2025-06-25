import os
import sys
import time
import torch
import argparse
import logging
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PIL
from PIL import Image
from efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite
from utils.train_utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_parameters

# 기본 설정 동일
CROP_PADDING = 32
MEAN_RGB = [0.498, 0.498, 0.498]
STDDEV_RGB = [0.502, 0.502, 0.502]

class DataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def next(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data[0], data[1]

def parse_device(device_str):
    device_str = device_str.lower()
    if device_str == 'cpu':
        return torch.device('cpu')
    elif device_str.isdigit():
        return torch.device(f'cuda:{device_str}')
    else:
        raise ValueError(f"Invalid device string: {device_str}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='efficientnet_lite0')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_resume', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument('--save', type=str, default='./models')
    parser.add_argument('--save_epoch_interval', type=int, default=1, help='save model interval')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    return parser.parse_args()

def main():
    args = get_args()

    os.makedirs(args.save, exist_ok=True)
    os.makedirs('./log', exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format='[%(asctime)s] %(message)s', datefmt='%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join('./log', f'train-{int(time.time())}.log'))
    logging.getLogger().addHandler(fh)

    input_size = efficientnet_lite_params[args.model_name][2]
    device = parse_device(args.device)
    use_gpu = (device.type == 'cuda') and torch.cuda.is_available()
    if use_gpu and 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = torch.device('cuda:0')

    train_dataset = datasets.ImageFolder(args.train_dir, transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_RGB, STDDEV_RGB),
    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=use_gpu)
    train_dataprovider = DataIterator(train_loader)

    val_dataset = datasets.ImageFolder(args.val_dir, transforms.Compose([
        transforms.Resize(input_size + CROP_PADDING, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_RGB, STDDEV_RGB),
    ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=200, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_gpu)
    val_dataprovider = DataIterator(val_loader)

    model = build_efficientnet_lite(args.model_name, args.num_classes)
    optimizer = torch.optim.SGD(get_parameters(model), lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if use_gpu:
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        criterion = criterion.to(device)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0)
        logging.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    if args.eval:
        checkpoint = torch.load(args.eval_resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        evaluate(model, device, val_dataprovider, criterion, epoch='Eval')
        return

    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        train_acc, train_loss = train(model, device, train_dataprovider, criterion, optimizer, epoch, args)
        val_acc, val_loss = evaluate(model, device, val_dataprovider, criterion, epoch)

        scheduler.step()

        # 1) resume 용 state_dict + optimizer + scheduler 상태 모두 저장 (overwrite)
        resume_save_path = os.path.join(args.save, f"{args.model_name}.pt")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, resume_save_path)

        # 2) 지정된 주기에 따라 epoch 별 모델 저장 (inference 용)
        if (epoch + 1) % args.save_epoch_interval == 0:
            epoch_save_path = os.path.join(args.save, f"{args.model_name}_epoch{epoch+1}.pt")
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, epoch_save_path)

        # 3) best model 갱신 및 저장 (val accuracy 기준)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_save_path = os.path.join(args.save, f"{args.model_name}_best.pt")
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, best_save_path)
            logging.info(f"Best model updated at epoch {epoch+1} with val ACC: {val_acc:.2f}%")

        log_str = f"Epoch {epoch+1}/{args.epochs} | Train ACC: {train_acc:.2f}%, Loss: {train_loss:.4f} | Val ACC: {val_acc:.2f}%, Loss: {val_loss:.4f}"
        logging.info(log_str)

def train(model, device, data_iter, criterion, optimizer, epoch, args):
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0

    for step in range(len(data_iter.dataloader)):
        data, target = data_iter.next()
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output, 1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        total_loss += loss.item() * target.size(0)

    acc = correct / total * 100
    loss = total_loss / total
    return acc, loss

def evaluate(model, device, data_iter, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(len(data_iter.dataloader)):
            data, target = data_iter.next()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item() * target.size(0)

    acc = correct / total * 100
    loss = total_loss / total
    return acc, loss

if __name__ == '__main__':
    main()

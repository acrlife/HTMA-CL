"""Val HTHA-CL on CIFAR10/CIFAR100."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from model import *
from timm.models import *
from cifar_utils import progress_bar
from timm.models import create_model,load_checkpoint

CUDA_VISIBLE_DEVICES=0,1

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='cifar10 or cifar100')
parser.add_argument('--data', type=str, default='D:/data',
                    help='datasets dictionary')
parser.add_argument('--b', type=int, default=16,
                    help='batch size')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num-classes', type=int, default=100, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--model', default='htha_14', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--img-size', type=int, default=384, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--rat', type=float, default=0.1,help='CS sampling ratio.')
parser.add_argument('--blocksize', type=int, default=32,help='Patch size')
parser.add_argument('--eval_checkpoint', default=r"D:\checkpoint\HTHA\cifar100\cifar100_384_r0.1_0.001_0.0001_86.68.pth",
                    type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
args = parser.parse_args()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
    transform_test = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset=='cifar10':
        print("Use cifar10")
        args.num_classes = 10
        testset = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=transform_test)

    elif args.dataset=='cifar100':
        print("Use cifar100")
        args.num_classes = 100
        testset = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=transform_test)
    else:
        print('Please use cifar10 or cifar100 dataset.')

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=2)

    print('==> Building model..')
    net = create_model(
            args.model,
            pretrained=args.pretrained,  # default:False
            num_classes=args.num_classes,
            checkpoint_path=args.initial_checkpoint,
            img_size=args.img_size,
            cs_ratio=args.rat,
            blocksize=args.blocksize
        )

    net = net.to(device)
    if device == 'cuda' and args.num_gpu >1:
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.num_gpu >=2:
        load_checkpoint(net.module, args.eval_checkpoint, args.model_ema)
    else:
        load_checkpoint(net, args.eval_checkpoint, args.model_ema)

    print('Load checkpoint from: ', args.eval_checkpoint)
    val_metrics = validate(net, testloader, criterion,device)
    print(f"Top-1 accuracy of the model is: {val_metrics:.2f}%")

def validate(model, loader, loss_fn,device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            print()
    acc = 100. * correct / total
    return acc

if __name__ == '__main__':
    main()


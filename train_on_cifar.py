"""Train HTMA-CL on CIFAR10/CIFAR100."""
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
from timm.models import create_model

from utils import load_for_transfer_learning
CUDA_VISIBLE_DEVICES=0,1

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='cifar10 or cifar100')
parser.add_argument('--data', type=str, default='D:/data',
                    help='datasets dictionary')
parser.add_argument('--save-path', type=str, default='./checkpoint',
                    help='datasets dictionary')
parser.add_argument('--b', type=int, default=64,
                    help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num-classes', type=int, default=100, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--model', default='htma_14', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--img-size', type=int, default=384, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--rat', type=float, default=0.1,help='CS sampling ratio.')
parser.add_argument('--blocksize', type=int, default=32,help='Patch size')
# Transfer learning
parser.add_argument('--transfer-learning', default=True,
                    help='Enable transfer learning')
parser.add_argument('--transfer-model', type=str, default="",
                    help='Path to pretrained model for transfer learning')
parser.add_argument('--transfer_ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')

args = parser.parse_args()
best_acc = 0  # best test accuracy

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset=='cifar10':
        print("Use cifar10")
        args.num_classes = 10
        trainset = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=transform_test)

    elif args.dataset=='cifar100':
        print("Use cifar100")
        args.num_classes = 100
        trainset = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=transform_test)
    else:
        print('Please use cifar10 or cifar100 dataset.')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=2)

    # print(f'learning rate:{args.lr}, weight decay: {args.wd}')
    # create T2T-ViT Model
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

    if args.transfer_learning:
        print('transfer learning, load pretrained model')
        load_for_transfer_learning(net, args.transfer_model, use_ema=False, strict=False, num_classes=args.num_classes)

    net = net.to(device)
    if device == 'cuda'and args.num_gpu >1:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    # set optimizer
    if args.transfer_learning:
        print('Set different lr for the sample module, backbone and classifier(head).')
        parameters = [{'params': net.module.sample.parameters()},
                      {'params': net.module.tokens_to_token.parameters(),'lr': args.transfer_ratio * args.lr},
                      {'params': net.module.blocks.parameters(), 'lr': args.transfer_ratio * args.lr},
                      {'params': net.module.head.parameters()},
        ]
    else:
        parameters = net.parameters()

    optimizer = optim.SGD(parameters, lr=args.lr,
                          momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'state_dict': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(f'{args.save_path}_{args.dataset}_384_{args.rat}'):
                os.makedirs(f'{args.save_path}_{args.dataset}_384_{args.rat}',exist_ok=True)
            torch.save(state, f'{args.save_path}_{args.dataset}_384_{args.rat}/ckpt_{acc}.pth')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+60):
        train(epoch)
        test(epoch)
        scheduler.step()

if __name__ == '__main__':
    main()


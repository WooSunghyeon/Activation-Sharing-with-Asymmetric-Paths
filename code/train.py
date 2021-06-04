import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import json
import os
import argparse

from convnet import *
from resnet import *
from utils import progress_bar
import numpy as np
import math
#%%
parser = argparse.ArgumentParser(description='Activation Sharing with Asymmetric Paths')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--batch-size', type=int, 
                    default=128, help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, 
                    default=100, help='input batch size for teset (default: 100)')
parser.add_argument('--epochs', type=int, default=200,  
                    help='number of epochs to train (default: 200)')
parser.add_argument('--dataset', type= str ,choices = ['mnist', 'cifar10', 'cifar100', 'svhn'], 
                    default='cifar10', help='choose dataset (default: cifar10)')
parser.add_argument('--model', type= str ,choices = ['convnet', 'resnet18_not','resnet18' ,'resnet34'], 
                    default='convnet', help='choose architecture (default: convnet)')
parser.add_argument('--feedback', type=str, 
                    choices = ['bp', 'fa', 'dfa', 'asap','asap_k4'],
                    default='bp',  help='feedback to use (default: bp)')
parser.add_argument('--aug', action = 'store_true', default = False, 
                    help = 'data augmentataion with random crop, horizontalflip (default : False)')
parser.add_argument('--wt', action = 'store_true', default = False
                    , help = 'activation sharing with transposed weight (default : False)')
parser.add_argument('--optimizer', type=str, choices = ['sgd', 'adam'], default = 'sgd'
                    , help = 'choose optimizer (default : sgd)')
parser.add_argument('--device', type= int, default = 0, help='device_num')

def main():
    args = parser.parse_args()
    device = args.device
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    # All data licensed under CC-BY-SA.
    dataset = args.dataset
    if dataset == 'mnist':
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.test_batch_size, shuffle=False)
    
    elif dataset == 'svhn':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        trainloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'train',download=True, transform=transform),
                                                  batch_size=args.batch_size,
                                                  shuffle=True)

        testloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'test',download=True, transform=transform),
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False)
        
    elif dataset == 'cifar10':
        if args.aug :
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else: 
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
                
    elif dataset == 'cifar100':
        if args.aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    
    # Set the model               
    learning_kwargs = {'dataset' : args.dataset,
                    'feedback' : args.feedback,
                    'wt' : args.wt}
   
    resnet_learning_kwargs = {'dataset' : args.dataset,
                               'feedback' : args.feedback,
                               'model' : args.model,
                               'wt' : args.wt}

    save_file_name = args.dataset + '_' + args.model + '_' + args.feedback
    print(learning_kwargs)
    
    if args.model == 'convnet':
        net = convnet(**learning_kwargs)
    else:
        if args.feedback == 'asap' or args.feedback == 'asap_k4':
            net = resnet_asap(**resnet_learning_kwargs)
        else:
            net = resnet(**resnet_learning_kwargs)
        
    net = net.to(device)
    print(net)
    
    # Decide optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
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
            
            p  = torch.rand(1).item()
            if p > 0 :
                bp = False
            else:
                bp = True
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
            
        return 100.*correct/total, train_loss
        
    def test(epoch, best_acc):
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
    
        acc = 100.*correct/total

        return acc, test_loss
        
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_acc, train_loss = train(epoch)
        test_acc, test_loss = test(epoch, best_acc)
        scheduler.step() 
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + save_file_name + '.pth')
            best_acc = test_acc
            print('best accuracy : ' + str(round(best_acc, 2)), 'epoch : ' + str(epoch) + '\n')
        
if __name__ == '__main__':
    main()      

#%%

from __future__ import print_function

# basic functions
import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# torch functions
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# local functions
from models import *
from utils import inceptionScore
from resnet import PreActResNet18

#--------------------------------------------------------------------
# input arguments
parser = argparse.ArgumentParser(description='BGF')
parser.add_argument('--dataset', default='mnist', help='mnist | fashionmnist')
parser.add_argument('--dataroot', default='mnist', help='path to dataset')

parser.add_argument('--gpuDevice', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

parser.add_argument('--nz', type=int, default=128, help='size of the latent vector')
parser.add_argument('--ngf', type=int, default=128)

parser.add_argument('--cuda', action='store_true',  help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='Results/MNIST/checkpoint/KL-mnist-0-ckpt.t7', help='path to netG')
parser.add_argument('--resnet', default='./checkpoint/resnet18-mnist-ckpt.t7', help='path to resnet for IS score')
parser.add_argument('--outf', default='./Results/eval', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=10, help='manual seed')

opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


if opt.dataset == 'mnist':
    nc = 1
    nclass = 10

elif opt.dataset == 'fashionmnist':
    nc = 1
    nclass = 10

elif opt.dataset == 'cifar10':
    nc = 3
    nclass = 10

else:
    raise NameError
    
is_score = []
    
device = torch.device('cuda:2' if torch.cuda.is_available() and not opt.cuda else 'cpu')  
    
netG = G_resnet(nc, opt.ngf, opt.nz)

netG.apply(weights_init)
netG.to(device)

state = torch.load(opt.netG)
netG.load_state_dict(state['netG'])
    
    

netIncept = PreActResNet18(nc)
netIncept.to(device)

if torch.cuda.is_available() and not opt.cuda:
    checkpoint = torch.load(opt.resnet)
    netIncept.load_state_dict(checkpoint['net'])

else:
    checkpoint = torch.load(inputf, map_location=lambda storage, loc: storage)
    netIncept.load_state_dict(checkpoint['net'])


# inception score
is_score.append(inceptionScore(netIncept, netG, device, opt.nz, nclass))
print('Inception Score is: %.4f' % (is_score[-1]))

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
from utils import *
from resnet import *

#--------------------------------------------------------------------
# input arguments
parser = argparse.ArgumentParser(description='BGF')
parser.add_argument('--divergence', '-div', type=str, default='KL', help='KL | logd | JS | Jeffrey')
parser.add_argument('--dataset', default='mnist', help='mnist | fashionmnist')
parser.add_argument('--dataroot', default='mnist', help='path to dataset')

parser.add_argument('--gpuDevice', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='input image size')

parser.add_argument('--nz', type=int, default=128, help='size of the latent vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)

parser.add_argument('--nEpoch', type=int, default=10, help='maximum Outer Loops')
parser.add_argument('--nDiter', type=int, default=1, help='number of D update')
parser.add_argument('--nPiter', type=int, default=20, help='number of particle update')
parser.add_argument('--nProj', type=int, default=20, help='number of G projection')
parser.add_argument('--nPool', type=int, default=20, help='times of batch size for particle pool')
parser.add_argument('--period', type=int, default=5, help='period of saving ckpts') 

parser.add_argument('--eta', type=float, default=0.5, help='learning rate for particle update')
parser.add_argument('--lrg', type=float, default=0.0001, help='learning rate for G, default=0.0001')
parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate for D, default=0.0001')
parser.add_argument('--lre', type=float, default=0.0001, help='learning rate for E, default=0.0001')

parser.add_argument('--decay_g', type=bool, default=True, help='lr_g decay')
parser.add_argument('--decay_e', type=bool, default=True, help='lr_e decay')
parser.add_argument('--decay_d', type=bool, default=True, help='lr_d decay')

parser.add_argument('--cuda', action='store_true',  help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--outf', default='./Results/MNIST', help='folder to output images and model checkpoints')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--start_save', type=int, default=800)
parser.add_argument('--manualSeed', type=int, default=10, help='manual seed')
parser.add_argument('--increase_nProj', type=bool, default=True, help='increase the projection times')

opt = parser.parse_args()


# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuDevice

try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.outf + '/gene')
except OSError:
    pass

try:
    os.mkdir('./projection_loss')
except:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
poolSize = opt.batchSize * opt.nPool

train_transforms = transforms.Compose([
                   transforms.Resize(opt.imageSize),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, ), (0.5, )),
                                     ])
if opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 1
    nclass = 10

elif opt.dataset == 'fashionmnist':
    dataset = dset.FashionMNIST(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 1
    nclass = 10

elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 3
    nclass = 10

else:
    raise NameError

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
p_dataloader = torch.utils.data.DataLoader(dataset, batch_size=poolSize,
                                         shuffle=True, num_workers=int(opt.workers))

n_samples = 10000
if (n_samples == None):
    n_samples = len(dataset)
elif n_samples < 0:
    n_samples = min(50000, len(dataset))
else:
    n_samples = min(n_samples, len(dataset))
data_loader2 = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True, num_workers=int(opt.workers))


device = torch.device('cuda:1' if torch.cuda.is_available() and not opt.cuda else 'cpu')
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
eta = float(opt.eta)
dataset = str(opt.dataset)

# nets
netD = D_resnet3(nc, ndf)
netG = G_resnet(nc, ngf, nz)
netE = Encoder(nz, True)


netG.apply(weights_init)
netG.to(device)
netD.apply(weights_init)
netD.to(device)
netE.apply(weights_init)
netE.to(device)
print('#-----------GAN initializd-----------#')

if opt.resume:
    assert os.path.isdir('./newresults4_49/checkpoint'), 'Error: no checkpoint directory found!'
    state = torch.load('./newresults4_49/checkpoint/%s-%s-%s-ckpt.t7' % (opt.divergence, opt.dataset, str(opt.resume_epoch)))
    netG.load_state_dict(state['netG'])
    netD.load_state_dict(state['netD'])
    netE.load_state_dict(state['netE'])
    start_epoch = state['epoch'] + 1
    is_score = state['is_score']
    best_is = state['best_is']
    fid_score = state['fid_score']
    best_fid = state['best_fid']
    loss_G = state['loss_G']
    optim_D = state['optim_D']
    optim_G = state['optim_G']
    optim_E = state['optim_E']
    print('#-----------Resumed from checkpoint-----------#')

else:
    start_epoch = 0
    is_score = []
    fid_score = []
    best_is = 0.0
    best_fid = 0.0

netIncept = PreActResNet18(nc)
netIncept.to(device)
#netIncept = torch.nn.DataParallel(netIncept)

if torch.cuda.is_available() and not opt.cuda:
    checkpoint = torch.load('./checkpoint/resnet18-'+ dataset +'-ckpt.t7')
    netIncept.load_state_dict(checkpoint['net'])

else:
    checkpoint = torch.load('./checkpoint/resnet18-' + dataset +' -ckpt.t7', map_location=lambda storage, loc: storage)
    netIncept.load_state_dict(checkpoint['net'])

# print('#------------Classifier load finished------------#')



z_b = torch.FloatTensor(opt.batchSize, nz, 1, 1).to(device)
img_b = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
p_z = torch.FloatTensor(poolSize, nz).to(device)
p_img = torch.FloatTensor(poolSize, nc, opt.imageSize, opt.imageSize).to(device)
p_lant = torch.FloatTensor(poolSize, nz, 1, 1).to(device)

show_z_b = torch.FloatTensor(64, nz).to(device)
eval_z_b = torch.FloatTensor(250, nz,1,1).to(device)

# set optimizer

optim_D = optim.RMSprop(netD.parameters(), lr=opt.lrd)
optim_G = optim.RMSprop(netG.parameters(), lr=opt.lrg)
optim_E = optim.RMSprop(netE.parameters(), lr=opt.lre)

if opt.dataset == 'mnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[400, 800], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[400, 800], gamma=0.5)
    scheduler_E = MultiStepLR(optim_E, milestones=[400, 800], gamma=0.5)

elif opt.dataset == 'fashionmnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[400, 800, 1200], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[400, 800, 1200], gamma=0.5)
    scheduler_E = MultiStepLR(optim_E, milestones=[400, 800, 1200], gamma=0.5)

elif opt.dataset == 'cifar10':
    scheduler_D = MultiStepLR(optim_D, milestones=[800, 1600, 2400], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[800, 1600, 2400], gamma=0.5)
    scheduler_E = MultiStepLR(optim_G, milestones=[800, 1600, 2400], gamma=0.5)

# set criterion
criterion_G = nn.MSELoss()

def my_norm(x):
    out = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    out = out/torch.norm(out, dim=1).reshape(out.shape[0],1)
    return out.reshape(x.shape)

def get_nProj_mfm(epoch):
    if epoch < 200:
        nProj_t = 5
    elif 199 < epoch < 1000:
        nProj_t = 10
    elif 999 < epoch < 1500:
        nProj_t = 15
    elif 1499 < epoch:
        nProj_t = 20

    return nProj_t

def get_nProj_cf(epoch):
    if epoch < 600:
        nProj_t = 5
    elif 599 < epoch < 2000:
        nProj_t = 10
    elif 1999 < epoch < 3000:
        nProj_t = 15
    elif 2999 < epoch:
        nProj_t = 20

    return nProj_t

def get_nProj_t(epoch):
    if opt.dataset == 'mnist' or 'fashionmnist':
        nProj_t = get_nProj_mfm(epoch)
    elif opt.dataset == 'cifar10':
        nProj_t = get_nProj_cf(epoch)
    else:
        raise NameError

    return nProj_t


def get_eta_mfm(epoch):
    if epoch < 200:
        eta = 30
    elif 199 < epoch < 1000:
        eta = 3
    elif 999 < epoch < 1500:
        eta = 0.5
    elif 1499 < epoch:
        eta = 0.5

    return eta


 
#--------------------------- main function ---------------------------#
real_show, _ = next(iter(dataloader))
vutils.save_image(real_show/ 2 + 0.5, './' + opt.outf + '/real-%s.png' % opt.dataset, padding=0)

for epoch in range(start_epoch, start_epoch + opt.nEpoch):    
    # decay lr
    if opt.decay_d:
        scheduler_D.step()
    if opt.decay_g:
        scheduler_G.step()
    if opt.decay_e:
        scheduler_E.step()
        
        
    eta = get_eta_mfm(epoch)

    # input_pool
    netD.train()
    netG.eval()
    netE.eval()
    p_z.normal_()
    #p_z = torch.randn(poolSize, nz, 1, 1).to(device)
    p_img.copy_(netG(p_z).detach())
    p_real_img = next(iter(p_dataloader))[0].to(device)
    p_lant.copy_(netE(p_real_img)[0].detach())
    
    

    for t in range(opt.nPiter): 

        for _ in range(opt.nDiter):
            
            # Update D
            netD.zero_grad()
            netG.zero_grad()
            netE.zero_grad()
            
            # real
            real_img, _ = next(iter(dataloader))
            img_b.copy_(real_img.to(device))
            z_b_idx = random.sample(range(poolSize), img_b.shape[0])
            z_b.copy_(p_lant[z_b_idx])
            real_D_err = torch.log(1 + torch.exp(-netD(p_real_img[z_b_idx], z_b))).mean()
            real_D_err.backward()

            # fake
            z_b_idx2 = random.sample(range(poolSize), opt.batchSize)
            img_b.copy_(p_img[z_b_idx2])
            #z_b.copy_(torch.randn(img_b.shape[0], nz, 1, 1).to(device))
            p_z_idx2 = p_z[z_b_idx2]
            fake_D_err = torch.log(1 + torch.exp(netD(img_b, p_z_idx2.reshape(opt.batchSize, 128, 1, 1)))).mean()
            fake_D_err.backward()
            
            optim_D.step()
            optim_G.step()
            optim_E.step()

        # update particle pool            
        p_img_t = p_img.clone().to(device)
        p_img_t.requires_grad_(True)
        if p_img_t.grad is not None:
            p_img_t.grad.zero_()
            
        p_lant_t = p_lant.clone().to(device)
        p_lant_t.requires_grad_(True)
        if p_lant_t.grad is not None:
            p_lant_t.grad.zero_()
        fake_D_score = netD(p_img_t, p_z.reshape(poolSize, 128, 1, 1))
        real_D_score = netD(p_real_img, p_lant_t)

        # set s(x)
        if opt.divergence == 'KL':
            s = torch.ones_like(fake_D_score.detach())
            s2 = 1 / real_D_score.detach().exp()

        elif opt.divergence == 'logd':
            s = 1 / (1 + fake_D_score.detach().exp())
            s2 = 1 / (real_D_score.detach().exp() + real_D_score.detach().exp() * real_D_score.detach().exp())
            
        elif opt.divergence == 'JS':
            s = 1 / (1 + 1 / fake_D_score.detach().exp())
            s2 = 1 / (1 + real_D_score.detach().exp())
            
        elif opt.divergence == 'Jeffrey':
            s = 1 + fake_D_score.detach().exp()
            s2 = 1 + 1 / real_D_score.detach().exp()

        else:
            raise NameError
        s.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).expand_as(p_img_t)
        s2.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).expand_as(p_lant_t)
        fake_D_score.backward(torch.ones(len(p_img_t)).to(device))
        real_D_score.backward(torch.ones(len(p_lant_t)).to(device))       
        p_img = torch.clamp(p_img + eta * s * p_img_t.grad, -1, 1)
        p_lant = torch.clamp(p_lant - eta * s2 * p_lant_t.grad, -1, 1)

    # update G
    netG.train()
    netE.train()
    netD.eval()
    poolset = poolSet(p_z.cpu(), p_img.cpu(), p_real_img.cpu(), p_lant.cpu())
    poolloader = torch.utils.data.DataLoader(poolset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

    loss_G = []

    # set nProj_t
    if opt.increase_nProj:
        nProj_t = get_nProj_t(epoch)
    else:
        nProj_t = opt.nProj

    for _ in range(nProj_t):

        loss_G_t = []
        for _, data_ in enumerate(poolloader, 0):
            netG.zero_grad()

            input1_, target1_, input2_, target2_ = data_
            pred1_ = netG(input1_.to(device))
            pred2_ = netE(input2_.to(device))[0]
            loss = criterion_G(pred1_, target1_.to(device)) + criterion_G(pred2_, target2_.to(device))
            loss.backward()

            optim_G.step()
            optim_E.step()
            loss_G_t.append(loss.detach().cpu().item())

        loss_G.append(np.mean(loss_G_t))
        
    vutils.save_image(target1_/ 2 + 0.5 , './' + opt.outf + '/particle-%s-%s-%s-%s.png' 
                          % (str(epoch).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), padding=0)
    vutils.save_image(netG(netE(input2_.to(device))[0].squeeze(3).squeeze(2)) / 2 + 0.5, './' + opt.outf + '/recon-%s-%s-%s-%s.png' 
                          % (str(epoch).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), padding=0)
    vutils.save_image(input2_ / 2 + 0.5, './' + opt.outf + '/recon-%s-%s-%s-%s-ori.png' 
                          % (str(epoch).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), padding=0)
    print('Epoch(%s/%s)%d: %.4fe-4 | %.4fe-4 | %.4f | %.4f |%.4fe-4 | %.4fe-4 '
          % (opt.divergence, opt.dataset, epoch, real_D_err*10000,fake_D_err*10000, p_img_t.grad.norm(p=2), p_lant_t.grad.norm(p=2), 
             real_D_score.mean(), fake_D_score.mean()))
    
    #-----------------------------------------------------------------
    if epoch % opt.period == 0:
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(loss_G, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Projection Loss')
        plt.legend()
        fig.savefig('./projection_loss/projection' + str(epoch).zfill(4) + '.png')
        plt.close()

        # show image
        netG.eval()
        show_z_b.normal_()
        fake_img = netG(show_z_b)
        vutils.save_image(fake_img.detach().cpu() / 2 + 0.5, './' + opt.outf + '/fake-%s-%s-%s-%s.png' 
                          % (str(epoch).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), padding=0)

        # inception score
        is_score.append(inceptionScore(netIncept, netG, device, nz, nclass))
        print('[%d] Inception Score is: %.4f' % (epoch, is_score[-1]))
        best_is = max(is_score[-1], best_is)

        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(opt.period * (np.arange(epoch//opt.period + 1)), is_score, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Inception Score')
        plt.legend()
        fig.savefig('./' + opt.outf + '/IS-%s-%s.png' % (opt.divergence, opt.dataset))
        plt.close()

        if best_is == is_score[-1]:
            print('Save the best Inception Score: %.4f' % is_score[-1])
        else:
            pass

    if epoch % 50 == 0:
        try:
            os.makedirs(opt.outf + '/checkpoint')
        except OSError:
            pass

        state = {
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'netE': netE.state_dict(),
            'optim_D': optim_D,
            'optim_E': optim_E,
            'optim_G': optim_G,
            'is_score': is_score,
            'loss_G': loss_G,
            'epoch': epoch,
            'best_is': best_is
            }
        torch.save(state, './' + opt.outf + '/checkpoint/%s-%s-%s-ckpt.t7' % (opt.divergence, opt.dataset, str(epoch)))


    #save IS
    if epoch % 50 == 0:
        dataframe = pd.DataFrame({'./' + opt.outf + '/IS-%s' % opt.divergence: is_score})
        dataframe.to_csv('./' + opt.outf + '/is-%s-%s.csv' % (opt.divergence, opt.dataset), sep=',')

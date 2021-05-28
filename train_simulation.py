from __future__ import print_function

# basic functions
import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

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

import torch.utils.data as DBGF
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl

# local functions
from models import Discriminator_FC_sim4

#--------------------------------------------------------------------
# input arguments
parser = argparse.ArgumentParser(description='BGF')
parser.add_argument('--divergence', '-div', type=str, default='KL', help='KL | logd | JS | Jeffrey')
parser.add_argument('--dataset', default='simulation', help='simulation')
parser.add_argument('--dataroot', default='simulation', help='path to dataset')

parser.add_argument('--gpuDevice', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='input image size')

parser.add_argument('--nz', type=int, default=1024, help='size of the latent vector')
parser.add_argument('--ngf', type=int, default=1024)
parser.add_argument('--ndf', type=int, default=1024)

parser.add_argument('--nEpoch', type=int, default=500, help='maximum Outer Loops')
parser.add_argument('--nDiter', type=int, default=1, help='number of D update')
parser.add_argument('--nPiter', type=int, default=20, help='number of particle update')
parser.add_argument('--nProj', type=int, default=20, help='number of G projection')
parser.add_argument('--nPool', type=int, default=20, help='times of batch size for particle pool')
parser.add_argument('--period', type=int, default=5, help='period of saving ckpts') 

parser.add_argument('--eta', type=float, default=0.5, help='learning rate for particle update')
parser.add_argument('--lrg', type=float, default=0.0001, help='learning rate for G, default=0.0001')
parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate for D, default=0.0001')
parser.add_argument('--decay_g', type=bool, default=True, help='lr_g decay')
parser.add_argument('--decay_e', type=bool, default=True, help='lr_e decay')
parser.add_argument('--decay_d', type=bool, default=True, help='lr_d decay')

parser.add_argument('--cuda', action='store_true',  help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--outf', default='./Results/simulation', help='folder to output images and model checkpoints')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--start_save', type=int, default=800)
parser.add_argument('--manualSeed', type=int, default=10, help='manual seed')
parser.add_argument('--increase_nProj', type=bool, default=True, help='increase the projection times')

opt = parser.parse_args()
print(opt)

# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuDevice

try:
    os.makedirs(opt.outf)
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


f = 'MixtureGaussian_simRes/' 
Gaussian = np.loadtxt(f + 'ring3.txt', dtype = 'float')
MixtureGaussian = np.loadtxt(f + 'MixGaussian8_3.txt', dtype = 'float')

Gaussian = torch.from_numpy(Gaussian).float()
torch_Gaussian = DBGF.TensorDataset(Gaussian)

MixtureGaussian = torch.from_numpy(MixtureGaussian).float() 
torch_MixtureGaussian = DBGF.TensorDataset(MixtureGaussian)

dataloader = torch.utils.data.DataLoader(torch_Gaussian, batch_size=opt.batchSize, shuffle=True)
dataloader2 = torch.utils.data.DataLoader(torch_MixtureGaussian, batch_size=opt.batchSize, shuffle=True)
p_dataloader = torch.utils.data.DataLoader(torch_Gaussian, batch_size=poolSize,
                                         shuffle=True, num_workers=int(opt.workers))
p_dataloader2 = torch.utils.data.DataLoader(torch_MixtureGaussian, batch_size=poolSize,
                                         shuffle=True, num_workers=int(opt.workers))


device = torch.device('cuda:2' if torch.cuda.is_available() and not opt.cuda else 'cpu')
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
eta = float(opt.eta)
nc = 1


netD = Discriminator_FC_sim4(2, 2, 2)
netD.to(device)

print('#-----------GAN initializd-----------#')

if opt.resume:
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    state = torch.load('./checkpoint/GBGAN-%s-%s-%s-ckpt.t7' % (opt.divergence, opt.dataset, str(opt.resume_epoch)))
    netG.load_state_dict(state['netG'])
    netD.load_state_dict(state['netD'])
    netE.load_state_dict(state['netE'])
    start_epoch = state['epoch'] + 1
    is_score = state['is_score']
    best_is = state['best_is']
    loss_G = state['loss_G']
    print('#-----------Resumed from checkpoint-----------#')

else:
    start_epoch = 0
    mean_score = []
    var_score = []
    best_mean = 0.0
    best_var = 0.0
    mean2_score = []
    var2_score = []
    best2_mean = 0.0
    best2_var = 0.0 
    

# # print('#------------Classifier load finished------------#')
z_b = torch.FloatTensor(opt.batchSize, nz, opt.imageSize, 1).to(device)
img_b = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, 1).to(device)
p_z = torch.FloatTensor(poolSize, 2).to(device)
p_img = torch.FloatTensor(poolSize, 2).to(device)
p_lant = torch.FloatTensor(poolSize, 2).to(device)


# set optimizer
optim_D = optim.RMSprop(netD.parameters(), lr=opt.lrd)

if opt.dataset == 'simulation':
    scheduler_D = MultiStepLR(optim_D, milestones=[400, 800], gamma=0.5)
    

# set criterion
criterion_G = nn.MSELoss()

 
#--------------------------- main function ---------------------------#

for epoch in range(start_epoch, start_epoch + opt.nEpoch):    
    # decay lr
    if opt.decay_d:
        scheduler_D.step()

    # input_pool
    netD.train()
    p_real_img = next(iter(p_dataloader))[0].to(device)
    p_z = next(iter(p_dataloader2))[0].to(device)

    
    if epoch==0:
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as pl
        
        p_img.copy_(p_z.detach())
        p_lant.copy_(p_real_img.detach())
        
        mean_value = torch.mean(p_lant)
        var_value = torch.var(p_lant) + mean_value*mean_value
        
        mean2_value = torch.mean(p_img)
        var2_value = torch.var(p_img) + mean2_value*mean2_value
        
        plt.style.use('classic')
        ax = sns.kdeplot(np.array(p_lant.cpu()), shade = True, cmap = "PuBu", bw = 0.1)
        ax.patch.set_facecolor('white')
        ax.collections[0].set_alpha(0)
        ax.set_xlabel('$Y_0$', fontsize = 15)
        ax.set_ylabel('$Y_1$', fontsize = 15)
        pl.xlim(-2, 2)
        pl.ylim(-2, 2)
        pl.savefig(opt.outf + '/Gau_ori.png', dpi=300)
        pl.show()

        plt.style.use('classic')
        ax = sns.kdeplot(np.array(p_img.cpu()), shade = True, cmap = "PuBu", bw = 0.1)
        ax.patch.set_facecolor('white')
        ax.collections[0].set_alpha(0)
        ax.set_xlabel('$Y_0$', fontsize = 15)
        ax.set_ylabel('$Y_1$', fontsize = 15)
        pl.xlim(-2, 2)
        pl.ylim(-2, 2)
        pl.savefig(opt.outf + '/MixGau_ori.png', dpi=300)
        pl.show()
    

    for t in range(opt.nPiter): 

        for _ in range(opt.nDiter):
            
            # Update D
            netD.zero_grad()
            
            # real
            z_b_idx = random.sample(range(poolSize), img_b.shape[0])
            real_D_err = torch.log(1 + torch.exp(-netD(p_real_img[z_b_idx], p_lant[z_b_idx]))).mean()
            real_D_err.backward()

            # fake
            z_b_idx = random.sample(range(poolSize), opt.batchSize)
            fake_D_err = torch.log(1 + torch.exp(netD(p_img[z_b_idx], p_z[z_b_idx]))).mean()
            fake_D_err.backward()    
            
            optim_D.step()

        # update particle pool            
        p_img_t = p_img.clone().to(device)
        p_img_t.requires_grad_(True)
        if p_img_t.grad is not None:
            p_img_t.grad.zero_()
            
        p_lant_t = p_lant.clone().to(device)
        p_lant_t.requires_grad_(True)
        if p_lant_t.grad is not None:
            p_lant_t.grad.zero_()
        fake_D_score = netD(p_img_t, p_z)[:,0]
        real_D_score = netD(p_real_img, p_lant_t)[:,0]

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
        s.unsqueeze_(1).expand_as(p_img_t)                        
        s2.unsqueeze_(1).expand_as(p_lant_t)
        fake_D_score.backward(torch.ones(len(p_img_t)).to(device))
        real_D_score.backward(torch.ones(len(p_lant_t)).to(device))               
        p_img = torch.clamp(p_img + eta * s * p_img_t.grad, -1.2, 1.2)
        p_lant = torch.clamp(p_lant - eta * s2 * p_lant_t.grad, -1.2, 1.2)
        
        
    if epoch%5==0:
              
        mean_value = torch.mean(p_lant)
        var_value = torch.var(p_lant) + mean_value*mean_value
        
        mean2_value = torch.mean(p_img)
        var2_value = torch.var(p_img) + mean2_value*mean2_value
        
        mean_score.append(mean_value)
        var_score.append(var_value)
        mean2_score.append(mean2_value)
        var2_score.append(var2_value)
        
        dataframe = pd.DataFrame({'./' + opt.outf + '/mean-%s' % 'MixGau': mean_score})
        dataframe.to_csv('./' + opt.outf + '/mean-%s.csv' % ('MixGau'), sep=',')
        
        dataframe = pd.DataFrame({'./' + opt.outf + '/var-%s' % 'MixGau': var_score})
        dataframe.to_csv('./' + opt.outf + '/var-%s.csv' % ('MixGau'), sep=',')
        
        dataframe = pd.DataFrame({'./' + opt.outf + '/mean-%s' % 'Gau': mean2_score})
        dataframe.to_csv('./' + opt.outf + '/mean-%s.csv' % ('Gau'), sep=',')
        
        dataframe = pd.DataFrame({'./' + opt.outf + '/var-%s' % 'Gau': var2_score})
        dataframe.to_csv('./' + opt.outf + '/var-%s.csv' % ('Gau'), sep=',')
        
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(5 * (np.arange(epoch//5 + 1)), mean_score, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Mean')
        plt.legend()
        fig.savefig('./' + opt.outf + '/mean-%s.png' % ('MixGau'))
        plt.close()
        
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(5 * (np.arange(epoch//5 + 1)), var_score, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Sec Moment')
        plt.legend()
        fig.savefig('./' + opt.outf + '/var-%s.png' % ('MixGau'))
        plt.close()
        
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(5 * (np.arange(epoch//5 + 1)), mean2_score, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Mean')
        plt.legend()
        fig.savefig('./' + opt.outf + '/mean-%s.png' % ('Gau'))
        plt.close()
        
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(5 * (np.arange(epoch//5 + 1)), var2_score, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Sec Moment')
        plt.legend()
        fig.savefig('./' + opt.outf + '/var-%s.png' % ('Gau'))
        plt.close()
        
        plt.style.use('classic')
        ax = sns.kdeplot(np.array(p_lant.cpu()), shade = True, cmap = "PuBu", bw = 0.1)
        ax.patch.set_facecolor('white')
        ax.collections[0].set_alpha(0)
        ax.set_xlabel('$Y_0$', fontsize = 15)
        ax.set_ylabel('$Y_1$', fontsize = 15)
        pl.xlim(-2, 2)
        pl.ylim(-2, 2)
        pl.savefig(opt.outf + '/Gau_' + str(epoch) + '.png', dpi=300)
        pl.show()
        pl.close()

        plt.style.use('classic')
        ax = sns.kdeplot(np.array(p_img.cpu()), shade = True, cmap = "PuBu", bw = 0.1)
        ax.patch.set_facecolor('white')
        ax.collections[0].set_alpha(0)
        ax.set_xlabel('$Y_0$', fontsize = 15)
        ax.set_ylabel('$Y_1$', fontsize = 15)
        pl.xlim(-2, 2)
        pl.ylim(-2, 2)
        pl.savefig(opt.outf + '/MixGau_' + str(epoch) + '.png', dpi=300)
        pl.show()
        pl.close()   
    
    print('Epoch(%s/%s)%d: %.4fe-4 | %.4fe-4 | %.4f | %.4f |%.4fe-4 | %.4fe-4 '
          % (opt.divergence, opt.dataset, epoch, real_D_err*10000,fake_D_err*10000, p_img_t.grad.norm(p=2), p_lant_t.grad.norm(p=2), 
             real_D_score.mean(), fake_D_score.mean()))
    

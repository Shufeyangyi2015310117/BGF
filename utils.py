import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

import argparse
import numpy as np
from torch import cuda
from torch.autograd import Variable
import torch.nn.functional as F


class poolSet(Dataset):
    
    def __init__(self, p_z, p_img, p_real_img, p_latent):
        self.len = len(p_z)
        self.z_data = p_z
        self.img_data = p_img
        self.real_data = p_real_img
        self.latent_data = p_latent
    
    def __getitem__(self, index):
        return self.z_data[index], self.img_data[index], self.real_data[index], self.latent_data[index]
    
    def __len__(self):
        return self.len


def inceptionScore(net, netG, device, nz, nclass, batchSize=250, eps=1e-6):
    
    net.to(device)
    netG.to(device)
    net.eval()
    netG.eval()

    pyx = np.zeros((batchSize*200, nclass))

    for i in range(200):

        eval_z_b = torch.randn(batchSize, nz).to(device)
        fake_img_b = netG(eval_z_b)
        
        pyx[i*batchSize: (i+1)*batchSize] = F.softmax(net(fake_img_b).detach(), dim=1).cpu().numpy()

    py = np.mean(pyx, axis=0)
    
    kl = np.sum(pyx * (np.log(pyx+eps) - np.log(py+eps)), axis=1)

    kl = kl.mean()
    
    return np.exp(kl)




def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
            
            
            
def return_size48_array_cpu(stl_path):
    batchsize = 1000
    stl_96_cpu = np.load(stl_path).astype(np.float32)
    #print(stl_96_cpu.shape)
    for n in range(0, 100000//batchsize):
        print(n*batchsize, (n+1)*batchsize)
        stl_96_gpu = Variable(stl_96_cpu[n*batchsize:(n+1)*batchsize].cuda())
        #print(stl_96_gpu.shape)
        x = F.average_pooling_2d(stl_96_gpu, 2).data
        if n==0:
            stl_48_cpu = x.cpu()
        else:
            stl_48_cpu = np.concatenate([stl_48_cpu, x.cpu()], axis=0)
    return stl_48_cpu

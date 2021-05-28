import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import utils
import math
from torch.nn import init
from torch.nn import utils



class ResBlock_D(nn.Module):
    def __init__(self, nf, down=False, nc=3, first=False):
        super(ResBlock_D, self).__init__()

        self.nf = nf
        self.down = down
        self.nc = nc
        self.first = first
        nf_in = nc if first else nf

        self.relu1 = nn.ReLU(True)
        self.conv1 = sn(nn.Conv2d(nf_in, nf, 3, 1, 1, bias=False), n_power_iterations=5)
        self.relu2 = nn.ReLU(True)
        self.conv2 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False), n_power_iterations=5)

        self.conv_shortcut = sn(nn.Conv2d(nf_in, nf, 1, 1, 0, bias=False), n_power_iterations=5)

    def forward(self, x):
        out = x if self.first else self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.down:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)   
            shortcut = self.conv_shortcut(x)
            shortcut = F.avg_pool2d(shortcut, kernel_size=2, stride=2)

        else:
            shortcut = x

        out += shortcut

        return out

class D_resnet3(nn.Module):
    def __init__(self, nc, ndf):
        super(D_resnet3, self).__init__()
        self.nc = nc
        self.ndf = ndf

        self.block1 = ResBlock_D(ndf, True, nc, True)
        self.block2 = ResBlock_D(ndf, True)
        self.block3 = ResBlock_D(ndf)
        self.block4 = ResBlock_D(ndf)
        self.relu = nn.ReLU(True)
        self.linear = sn(nn.Linear(2*ndf, 1), n_power_iterations=5)
        
        
        self.infer_z = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 128, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        
        self.infer_joint = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        

    def forward(self, x, z):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.relu(out)
        out = out.sum(-1).sum(-1)  
        
        out2 = self.infer_z(z)
        
        out = self.infer_joint(torch.cat((out.view(-1, self.ndf, 1, 1), out2.view(-1, self.ndf, 1, 1)), 1))
        
        out = self.linear(out.view(-1, 2*self.ndf))
        return out.view(-1, 1).squeeze()   



class Encoder(nn.Module):

    def __init__(self, latent_size, noise=False):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

#         if noise:
#             self.latent_size *= 2
        self.main1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.main2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main4 = nn.Sequential(
            sn(nn.Conv2d(512, self.latent_size, 1, stride=1, bias=True), n_power_iterations=5),
#            nn.Tanh()
        )
        

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3)
        return output, x3.view(batch_size, -1), x2.view(batch_size, -1), x1.view(batch_size, -1)


class ResBlock_G(nn.Module):
    def __init__(self, nf, up=False):
        super(ResBlock_G, self).__init__()

        self.nf = nf
        self.up = up

        self.SubBlock1 = nn.Sequential(
                         nn.ReLU(True)
        )

        self.SubBlock2 = nn.Sequential(
                         sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False), n_power_iterations=5),
                         nn.BatchNorm2d(nf),
                         nn.ReLU(True),
                         sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False), n_power_iterations=5)
        )

        self.conv_shortcut = sn(nn.Conv2d(nf, nf, 1, 1, 0, bias=False), n_power_iterations=5)

    def forward(self, x):
        out = self.SubBlock1(x)

        if self.up:
            out = F.interpolate(out, scale_factor=2)
            shortcut = F.interpolate(x, scale_factor=2)
            shortcut = self.conv_shortcut(shortcut)

        else:
            shortcut = x

        out = self.SubBlock2(out)
        out += shortcut

        return out


class G_resnet(nn.Module):
    def __init__(self, nc=3, ngf=128, nz=128):
        super(G_resnet, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz

        self.linear = sn(nn.Linear(nz, 16*ngf), n_power_iterations=5)
        self.block1 = ResBlock_G(ngf, True)
        self.block2 = ResBlock_G(ngf, True)
        self.block3 = ResBlock_G(ngf, True)
        self.block4 = nn.Sequential(
                      nn.ReLU(True),
                      sn(nn.Conv2d(ngf, nc, 3, 1, 1, bias=False), n_power_iterations=5),
                      nn.Tanh()
        )


    def forward(self, x):
        out = self.linear(x)
        out = self.block1(out.view(-1, self.ngf, 4, 4))
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        return out.view(-1, self.nc, 32, 32)
        
        
        
#-------------------- init ----------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)

    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)





class Discriminator_FC_sim4(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is a tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Discriminator_FC_sim4, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim


        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*z_dim, 16*z_dim),
            nn.LeakyReLU(0),
            torch.nn.Linear(16*z_dim, 16*z_dim),
            nn.LeakyReLU(0),
            torch.nn.Linear(16*z_dim, 16*h_dim),
            nn.LeakyReLU(0),
            torch.nn.Linear(16*h_dim, 1)
            )

#         utils.initialize_weights(self)

    def forward(self, input_x, input_z):
        return self.fc(torch.cat([input_x, input_z], 1)) 

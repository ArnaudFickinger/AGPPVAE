import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import scipy as sp
import os
import pdb
import pylab as pl

def f_act(x, act="elu"):
    if act == "relu":
        return F.relu(x)
    elif act == "elu":
        return F.elu(x)
    elif act == "linear":
        return x
    else:
        return None


class Conv2dCellDown(nn.Module):
    def __init__(self, ni, no, ks=3, act="elu"):
        super(Conv2dCellDown, self).__init__()
        self.act = act
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=2, padding=1)

    def forward(self, x):
        x = f_act(self.conv1(x), act=self.act)
        x = f_act(self.conv2(x), act=self.act)
        return x


class Conv2dCellUp(nn.Module):
    def __init__(self, ni, no, ks=3, act1="elu", act2="elu"):
        super(Conv2dCellUp, self).__init__()
        self.act1 = act1
        self.act2 = act2
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=1, padding=1)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2)
        x = f_act(self.conv1(x), act=self.act1)
        x = f_act(self.conv2(x), act=self.act2)
        return x


class Encoder(nn.Module):
    def __init__(self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0, ks=3):
        super(Encoder, self).__init__()
        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf
        self.gamma = gamma
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors
        self.dense_zm = nn.Linear(self.size_flat, zdim)
        self.dense_zs = nn.Linear(self.size_flat, zdim)

        # conv cells encoder
        self.econv = nn.ModuleList()
        cell = Conv2dCellDown(colors, nf, ks, act)
        self.econv += [cell]
        for i in range(steps - 1):
            cell = Conv2dCellDown(nf, nf, ks, act)
            self.econv += [cell]

    def forward(self, x):
        for ic, cell in enumerate(self.econv):
            x = cell(x)
        x = x.view(-1, self.size_flat)
        zm = self.dense_zm(x)
        zs = F.softplus(self.dense_zs(x))
        return zm, zs


class Decoder(nn.Module):
    def __init__(self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0, ks=3):
        super(Decoder, self).__init__()
        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf
        self.gamma = gamma
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors
        self.dense_dec = nn.Linear(zdim, self.size_flat)
        # conv cells decoder
        self.dconv = nn.ModuleList()
        for i in range(steps - 1):
            cell = Conv2dCellUp(nf, nf, ks, act1=act, act2=act)
            self.dconv += [cell]
        cell = Conv2dCellUp(nf, colors, ks, act1=act, act2="linear")
        self.dconv += [cell]

    def forward(self, x):
        x = self.dense_dec(x)
        x = x.view(-1, self.nf, self.red_img_size, self.red_img_size)
        for cell in self.dconv:
            x = cell(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0, ks=3):
        super(Discriminator, self).__init__()
        # conv cells discriminator
        self.disconv = nn.ModuleList()
        cell = Conv2dCellDown(colors, nf, ks, act)
        self.disconv += [cell]
        for i in range(steps - 1):
            cell = Conv2dCellDown(nf, nf, ks, act)
            self.disconv += [cell]
        cell = nn.Conv2d(nf, 1, 4)
        self.disconv += [cell]
        cell = nn.Sigmoid()
        self.disconv += [cell]

    def forward(self, x):
        for cell in self.disconv:
            x = cell(x)
        return x


class FaceVAE(nn.Module):
    def __init__(
        self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, gamma = 0
    ):

        super(FaceVAE, self).__init__()

        # store useful stuff
        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf
        self.gamma = gamma
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors
        ks = 3

        # define variance
        self.vy = nn.Parameter(torch.Tensor([vy]), requires_grad=False)

        # dense layers
        self.dense_zm = nn.Linear(self.size_flat, zdim)
        self.dense_zs = nn.Linear(self.size_flat, zdim)
        self.dense_dec = nn.Linear(zdim, self.size_flat)

        self.encode = Encoder()
        self.decode = Decoder()
        self.discrim =  Discriminator()

    def sample(self, x, eps):
        zm, zs = self.encode(x)
        z = zm + eps * zs
        return z


    def forward(self, x, eps):

        dreal = self.discrim(x)

        zm, zs = self.encode(x)
        z = zm + eps * zs
        xr = self.decode(z)
        
        mse = ((xr - x) ** 2).view(x.shape[0], self.K).mean(1)[:, None]

        abs = (torch.abs(xr - x)).view(x.shape[0], self.K).mean(1)[:, None]

        dfake = self.discrim(xr)

        kld = (
            -0.5 * (1 + 2 * torch.log(zs) - zm ** 2 - zs ** 2).sum(1)[:, None] / self.K
        )

        return kld, abs, dreal, dfake, mse


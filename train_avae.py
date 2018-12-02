import matplotlib
import sys

matplotlib.use("Agg")
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from avae import FaceVAE
import h5py
import scipy as sp
import os
import pdb
import logging
import pylab as pl
from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from callbacks import callback_avae
from data_parser import read_face_data, FaceDataset
from optparse import OptionParser
import logging
import pickle
import time
import numpy as np


parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="./data/faceplace/data_faces.h5",
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/avae7", help="output dir"
)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--filts", dest="filts", type=int, default=32, help="number of convol filters"
)
parser.add_option("--zdim", dest="zdim", type=int, default=256, help="zdim")
parser.add_option(
    "--vy", dest="vy", type=float, default=2e-3, help="conditional norm lik variance"
)
parser.add_option("--lr", dest="lr", type=float, default=2e-4, help="learning rate")
parser.add_option("--bs", dest="bs", type=int, default=64, help="batch size")
parser.add_option(
    "--epoch_cb",
    dest="epoch_cb",
    type=int,
    default=10,
    help="number of epoch by which a callback (plot + dump weights) is executed",
)
parser.add_option(
    "--epochs", dest="epochs", type=int, default=51, help="total number of epochs"
)
parser.add_option("--debug", action="store_true", dest="debug", default=False)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)


if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# output dir
wdir = os.path.join(opt.outdir, "weights")
fdir = os.path.join(opt.outdir, "plots")
if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(fdir):
    os.makedirs(fdir)

# copy code to output folder
export_scripts(os.path.join(opt.outdir, "scripts"))

# create logfile
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(opt.outdir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("opt = %s", opt)

# extract VAE settings and export
vae_cfg = {"nf": opt.filts, "zdim": opt.zdim, "vy": opt.vy}
pickle.dump(vae_cfg, open(os.path.join(opt.outdir, "vae.cfg.p"), "wb"))


def main():

    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    # define VAE and optimizer
    vae = FaceVAE(**vae_cfg).to(device)
    bce = nn.BCELoss(reduction='sum').to(device)

    # optimizer
    optimizer_Enc = optim.Adam(vae.encode.parameters(), lr=0.0003)
    optimizer_Dec = optim.Adam(vae.decode.parameters(), lr=0.0003)
    optimizer_Dis = optim.Adam(vae.discrim.parameters(), lr=0.00003)

    # load data
    img, obj, view = read_face_data(opt.data)  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    val_data = FaceDataset(img["val"], obj["val"], view["val"])
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt.bs, shuffle=False)

    history = {}
    for epoch in range(opt.epochs):

        # train and eval
        ht = train_ep(vae, bce, train_queue, optimizer_Enc ,optimizer_Dec ,optimizer_Dis, gamma=1)
        hv = eval_ep(vae, bce, val_queue, gamma = 1)
        smartAppendDict(history, ht)
        smartAppendDict(history, hv)
        logging.info(
            "epoch %d - train_mse: %f- train_abs: %f- train_kld: %f - train_loss_enc: %f- train_loss_dec: %f- train_loss_dis: %f - test_mse %f - test_abs %f - test_kld %f - test_loss_enc %f- test_loss_dec %f- test_loss_dis %f" % (epoch, ht["mse"], ht["abs"], ht["kld"], ht["loss_enc"], ht["loss_dec"], ht["loss_dis"],  hv["mse_val"], hv["abs_val"], hv["kld_val"], hv["loss_enc_val"],hv["loss_dec_val"],hv["loss_dis_val"])
        )
        

        # callbacks
        if epoch % opt.epoch_cb == 0:
            logging.info("epoch %d - executing callback" % epoch)
            wfile = os.path.join(wdir, "weights.%.5d.pt" % epoch)
            ffile = os.path.join(fdir, "plot.%.5d.png" % epoch)
            torch.save(vae.state_dict(), wfile)
            callback_avae(epoch, val_queue, vae, history, ffile, device)


def train_ep(vae, bce, train_queue, optimizer_Enc ,optimizer_Dec ,optimizer_Dis, gamma=1):

    rv = {}
    vae.train()

    for batch_i, data in enumerate(train_queue):
        
        batch_size = len(data[0])

        ones_label = Variable(torch.ones(batch_size)).to(device)
        zeros_label = Variable(torch.zeros(batch_size)).to(device)

        # forward
        y = data[0]
        eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
        y, eps = y.to(device), eps.to(device)
        kld, abs, dreal, dfake, mse = vae.forward(y, eps)

        real_loss = bce(dreal, ones_label)
        fake_loss = bce(dfake, zeros_label)

        gen_loss = bce(dfake, ones_label)

        loss_dis = real_loss + fake_loss
        loss_dec = gamma * abs.sum() + gen_loss
        loss_enc = (kld+abs).sum()

        optimizer_Dis.zero_grad()
        loss_dis.backward(retain_graph=True)
        optimizer_Dis.step()

        optimizer_Dec.zero_grad()
        loss_dec.backward(retain_graph=True)
        optimizer_Dec.step()

        optimizer_Enc.zero_grad()
        loss_enc.backward()
        optimizer_Enc.step()

        # sum metrics
        _n = train_queue.dataset.Y.shape[0]
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / float(_n))
        smartSum(rv, "abs", float(abs.data.sum().cpu()) / float(_n))
        smartSum(rv, "kld", float(kld.data.sum().cpu()) / float(_n))
        smartSum(rv, "loss_dis", float(loss_dis.data.cpu()) / float(_n))
        smartSum(rv, "loss_dec", float(loss_dec.data.cpu()) / float(_n))
        smartSum(rv, "loss_enc", float(loss_enc.data.cpu()) / float(_n))

    return rv


def eval_ep(vae, bce, val_queue, gamma):
    rv = {}
    vae.eval()

    with torch.no_grad():

        for batch_i, data in enumerate(val_queue):
            batch_size = len(data[0])

            # forward
            y = data[0]
            eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
            ones_label = Variable(torch.ones(batch_size)).to(device)
            zeros_label = Variable(torch.zeros(batch_size)).to(device)

            y, eps = y.to(device), eps.to(device)
            kld, abs, dreal, dfake, mse = vae.forward(y, eps)

            real_loss = bce(dreal, ones_label)
            fake_loss = bce(dfake, zeros_label)
            
            gen_loss = bce(dfake, ones_label)

            loss_dis = real_loss + fake_loss
            loss_dec = gamma * abs.sum() + gen_loss
            loss_enc = (kld + abs).sum()

            # sum metrics
            _n = val_queue.dataset.Y.shape[0]
            smartSum(rv, "mse_val", float(mse.data.sum().cpu()) / float(_n))
            smartSum(rv, "abs_val", float(abs.data.sum().cpu()) / float(_n))
            smartSum(rv, "kld_val", float(kld.data.sum().cpu()) / float(_n))
            smartSum(rv, "loss_dis_val", float(loss_dis.data.cpu()) / float(_n))
            smartSum(rv, "loss_dec_val", float(loss_dec.data.cpu()) / float(_n))
            smartSum(rv, "loss_enc_val", float(loss_enc.data.cpu()) / float(_n))

    return rv


if __name__ == "__main__":
    main()

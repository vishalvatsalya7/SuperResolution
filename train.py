import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import pickle
import os
import cv2
import torch.optim as optim
import torch.nn as nn
import PIL.Image as Image
from model import Model
from sklearn.model_selection import train_test_split
import argparse

#1356, 2040, 3 original HR image'dim
#L1 loss provide better convergence than L2 loss (Source : https://arxiv.org/pdf/1707.02921.pdf)

#for GPU usage
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--ni', type=int, help='number of color channels', default=3)
parser.add_argument('--nf', type=int, default=256, help='number of filters')
parser.add_argument('--n_resblocks', type=int, default=32, help='number of residual blocks')
parser.add_argument('--scale', type=int, default=4, help='by how much the image is to be scaled')
parser.add_argument('--epochs', type=int, default=1000, help='number of iterations')

opt = parser.parse_args()
model = Model(ni=opt.ni, nf=opt.nf, n_resblocks=opt.n_resblocks, scale=opt.scale)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model.cuda()

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

LQ_train, LQ_test, HQ_train,HQ_test = train_test_split(X,y,test_size=0.20,random_state=42)
# cv2.blur(HQ_test[2], (3,3)) for adding blur in images for training

def L1_loss(ip,op):
    loss = nn.L1Loss()
    return loss(ip, op)

def train(epoch):
    model.train()
    train_loss = 0
    for i, x in enumerate(LQ_train):
        x = np.expand_dims(x, axis=0)
        x = x.transpose(0, 3, 2, 1)
        data = torch.tensor(x).type(torch.FloatTensor)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = np.expand_dims(HQ_train[i], axis=0)
        target = target.transpose(0, 3, 2, 1)
        target = torch.tensor(target).type(torch.FloatTensor)
        target = target.to(device)
        loss = L1_loss(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(data), len(LQ_train),
                   100. * i / len(LQ_train),
                   loss.item() / len(data)))
        if i%20==0:
            torch.save(model.state_dict(), "Models_sr_3")
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(LQ_train)))


def main():
    for epoch in range(opt.epochs):
        train(epoch)



if __name__ == "__main__":
    main()

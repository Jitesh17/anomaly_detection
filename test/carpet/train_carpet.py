import os
import sys

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# import cv2
# from torch.autograd import Variable
from torchvision.utils import save_image

from src.lib import (augment, feature_score, get_dataloader, instance_score,
                     inverse_normalize, make_figure, make_full_figure, show_im, tensor2np)

img_size = 64*2
num_epochs = 300
checkpoint = 100
batch_size = 4
learning_rate = 0.001
threshold = 0.01
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean = std = [0.5]*3


base = ""
# base = "/home/jitesh/jg/anomaly_detection/"
train_num = 5
data_name = "carpet"  # "capsule"  # "carpet"
weight_dir_path = base + f"weights/{data_name}/{train_num}"
writer = SummaryWriter(log_dir=os.path.join(weight_dir_path, f"tb"))


dataloader, data_len = get_dataloader(base + f"data/mvtec/{data_name}/train/good", mode='val',
                            batch_size=batch_size, shuffle=True, num_workers=8, img_size=img_size)
testloader, test_len = get_dataloader(base + f"data/mvtec/{data_name}/train/good", mode='test',
                            batch_size=batch_size, shuffle=False, num_workers=8, img_size=img_size)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)


dense_dim = [16, img_size//2, img_size//2]
e1 = np.prod(dense_dim)
e2 = 1024
e3 = 128


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(                                 # 3 x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=8,
                      kernel_size=3, padding=1),  # 8 x 128 x 128
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16 x 64 x 64
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16 x 64 x 64
            nn.Flatten(),
            nn.Linear(e1, e2),
            # nn.Linear(e2, e3),
        )
        self.decoder = nn.Sequential(
            # nn.Linear(e3, e2),
            nn.Linear(e2, e1),
            View([-1, 16, img_size//2, img_size//2]),
            # 8 x 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2,
                               padding=0),  # 16 x 128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1,
                               padding=1),  # 3 x 128 x 128
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = 'cuda'
# model = Autoencoder().cpu()
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[280*50//batch_size, 280*100//batch_size], gamma=0.3)
def get_lr():
    for param_group in optimizer.param_groups:
        return param_group['lr']

# images, _ = next(iter(dataloader))
# grid = tv.utils.make_grid(images)
# writer.add_image('test', grid, 0)
# print(images.shape)
# print(images.shape)
# image = images.cpu().numpy()[0]
# writer.add_figure('fig1', make_figure(images[0]), 0)
for epoch in range(1, num_epochs+1):
    train_loss = 0.0

    for data in dataloader:
        images, _ = data
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()*images.size(0)
    fig, [fscore, iscore, thres] = make_full_figure(images, outputs, batch_size, img_size)
    writer.add_figure('Good Images', fig, epoch)
    # print(iscore)
    # writer.add_scalar('Score/I-Score/training', iscore, epoch)
    writer.add_scalar('Score/I-Score-threshold/training', thres, epoch)
    # writer.add_image('Figure', make_full_figure(images, outputs, batch_size), epoch)
    # print(images.shape)
    # image = images.cpu().numpy()[0]
    # # image = np.transpose(image, (1, 2, 0))
    # print(image.shape)
    # image = image.astype(np.uint8)
    # print(image.shape)
    # # image = np.asarray(Image.fromarray(image).convert("RGB"))
    # print(image.shape)
    # # grid = tv.utils.make_grid(images)
    # writer.add_image('Input', image, epoch, dataformats='CHW' )
    # grid = tv.utils.make_grid(outputs)
    # writer.add_image('Output', grid, epoch)
    # images = images.cpu().numpy()
    # writer.add_figure('Output_fig', tensor2np(grid.cpu().numpy()[0]), 0)
    train_loss = train_loss/len(dataloader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Learning Rate', get_lr(), epoch)
    writer.add_graph(model, images, 0)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, train_loss))
    
    if epoch%checkpoint == 0:
        torch.save(model.state_dict(), os.path.join(weight_dir_path, f'epoch_{epoch}.pth'))
        dataiter = iter(testloader)
        thres_list = []
        for i in range(int(np.ceil(test_len/batch_size))):
            images, labels = dataiter.next()
            images = images.to(device)
            output = model(images)
            images = images.cpu().numpy()
            output = output.view(-1, 3, img_size, img_size)
            output = output.detach().cpu().numpy()

            fscore = feature_score(images, output)
            tp = 100
            iscore = instance_score(fscore, tp)
            score = iscore
            threshold = np.percentile(score, tp)
            thres_list.append(threshold)
        threshold = max(thres_list)
        f = open(os.path.join(weight_dir_path, "threshold.txt"), "w+")
        f.write(f"{threshold}")
        f.close()

writer.close()

filename_pth = f'final_model.pth'
torch.save(model.state_dict(), os.path.join(weight_dir_path, filename_pth))


dataiter = iter(testloader)
thres_list = []
for i in range(int(np.ceil(test_len/batch_size))):
    images, labels = dataiter.next()
    images = images.to(device)
    output = model(images)
    images = images.cpu().numpy()
    output = output.view(-1, 3, img_size, img_size)
    output = output.detach().cpu().numpy()

    fscore = feature_score(images, output)
    tp = 100
    iscore = instance_score(fscore, tp)
    score = iscore
    threshold = np.percentile(score, tp)
    thres_list.append(threshold)
threshold = max(thres_list)
f = open(os.path.join(weight_dir_path, "threshold.txt"), "w+")
f.write(f"{threshold}")
f.close()

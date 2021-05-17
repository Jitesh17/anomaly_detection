
import os

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
from torchvision.utils import save_image

def augment(img_size, mean=[0.5]*3, std=[0.5]*3):
    aug_seq1 = A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Transpose(p=1.0),
    ], p=1.0)
    aug_seq2 = A.OneOf([
        # A.RandomGamma(always_apply=False, p=1.0, gamma_limit=(80, 120), eps=1e-07),
        A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.2, 0.2),
                                contrast_limit=(-0.2, 0.2), brightness_by_max=True),
    ], p=1.0)
    aug_seq3 = A.OneOf([
        A.RGBShift(always_apply=False, p=1.0, r_shift_limit=(-10, 10),
                g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
        A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-4, 4),
                            sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),

    ], p=1.0)
    aug_seq4 = A.OneOf([
        A.MultiplicativeNoise(always_apply=False, p=1.0, multiplier=(
            0.8999999761581421, 1.100000023841858), per_channel=True, elementwise=True),
        A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 7)),
        A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0)),
        A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),

    ], p=1.0)
    aug_seq = A.Compose([
        A.Resize(img_size, img_size),
        aug_seq1,
        aug_seq2,
        aug_seq3,
        aug_seq4,
        A.Normalize(mean=mean, std=std),
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return aug_seq

def feature_score(X_orig: np.ndarray, X_recon: np.ndarray) -> np.ndarray:
    fscore = np.power(X_orig - X_recon, 2)
    return fscore


def instance_score(fscore: np.ndarray, outlier_perc: float = 100.) -> np.ndarray:
    fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
    n_score_features = int(np.ceil(.01 * outlier_perc * fscore_flat.shape[1]))
    sorted_fscore = np.sort(fscore_flat, axis=1)
    sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
    iscore = np.mean(sorted_fscore_perc, axis=1)
    return iscore


def inverse_normalize(tensor, mean=[0.5]*3, std=[0.5]*3):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2np(img, mean=[0.5]*3, std=[0.5]*3):
    inv_tensor = inverse_normalize(torch.tensor(
        img, dtype=torch.float), mean=mean, std=std)
    img = inv_tensor
    # img = img / 2 + 0.5
    img = np.clip(img, 0., 1.)
    return np.transpose(img, (1, 2, 0))


def show_im(img):
    plt.imshow(tensor2np(img))


class MyDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', aug=None,
                 test_label: int = 1,
                 #  b_type_list: List[str] = ['b10', 'b11'],
                 img_size: int = 256, mean=[0.5]*3, std=[0.5]*3):
        # super().__init__()
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        # self.transform = transform
        self.test_label = test_label
        # self.b_type_list = b_type_list
        self.img_size = img_size
        if self.mode == 'train' or self.mode == 'val':
            # print(self.file_list)
            # if 'b00' in self.file_list[0]:
            # if b_type_list[0] in self.file_list[0]:
            #     self.label = 0
            # else:
            #     self.label = 1
            self.label = 0
        self.aug = aug
        self.val_aug_seq = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.file_list[idx])
        image = Image.open(img_path)
        big_side = max(image.size)
        new_im = Image.new("RGB", (big_side, big_side))
        new_im.paste(image)
        image = new_im
        if self.mode == 'train':
            image = self.aug(image=np.array(image))['image']
            # image = self.val_aug_seq(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return torch.tensor(image, dtype=torch.float), self.label
        elif self.mode == 'val':
            image = self.val_aug_seq(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return torch.tensor(image, dtype=torch.float), self.label
        else:
            image = self.val_aug_seq(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return torch.tensor(image, dtype=torch.float), self.file_list[idx]


def get_files(dirpath):
    return [os.path.join(dirpath, fn) for fn in os.listdir(dirpath)]


def get_dataloader(path, batch_size, mode, shuffle, num_workers, img_size, mean=[0.5]*3, std=[0.5]*3):
    files = get_files(path)
    dataset = MyDataset(files, "", mode=mode,
                        aug=augment(img_size, mean, std),
                        test_label=0, img_size=img_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, len(files)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma

    def forward(self, x):
        return x.view(*self.shape)


class Autoencoder(nn.Module):
    def __init__(self, img_size=128):
        super(Autoencoder, self).__init__()

        self.dense_dim = [16, img_size//2, img_size//2]
        self.e1 = np.prod(self.dense_dim)
        self.e2 = 1024
        self.e3 = 128
        self.encoder = nn.Sequential(                    # 3 x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=8,
                      kernel_size=3, padding=1),         # 8 x 128 x 128
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16 x 64 x 64
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          # 16 x 64 x 64
            nn.Flatten(),
            nn.Linear(self.e1, self.e2),                 # 1024
            # nn.Linear(self.e2, self.e3),
        )
        self.decoder = nn.Sequential(
            # nn.Linear(e3, e2),
            nn.Linear(self.e2, self.e1),
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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

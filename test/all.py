
# In[]
import os
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
# import cv2
# from torch.autograd import Variable
from torchvision.utils import save_image
img_size = 64*2
num_epochs = 100
batch_size = 4
learning_rate = 0.001
threshold = 0.01
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean = std = [0.5]*3


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
    # aug_seq1,
    # aug_seq2,
    # aug_seq3,
    # aug_seq4,
    A.Normalize(mean=mean, std=std),
    # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
base = ""
base = "/home/jitesh/jg/anomaly_detection/"
weight_dir_path = base + "weights/simple2"
writer = SummaryWriter(log_dir=os.path.join(weight_dir_path, "tb"))
transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
])
trainTransform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
])


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


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2np(img):
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
                 img_size: int = 256):
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
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261)),
            A.Normalize(mean=mean, std=std),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.file_list[idx])
        image = Image.open(img_path)
        big_side = max(image.size)
        # small_side = min(image.size)
        # printj.red(list(image.size)[0])
        # print(big_side)
        new_im = Image.new("RGB", (big_side, big_side))
        new_im.paste(image)
        image = new_im
        # x()
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
# In[]


def get_files(dirpath):
    return [os.path.join(dirpath, fn) for fn in os.listdir(dirpath)]


def get_dataloader(path, batch_size, mode, shuffle, num_workers):
    files = get_files(path)
    dataset = MyDataset(files, "", mode=mode,
                        aug=aug_seq,
                        test_label=0, img_size=img_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, len(files)


dataloader, data_len = get_dataloader(base + "data/washer/washer_ok", mode='train',
                            batch_size=batch_size, shuffle=True, num_workers=8)
testloader, test_len = get_dataloader(base + "data/washer/washer_ok", mode='test',
                            batch_size=batch_size, shuffle=False, num_workers=8)


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
            nn.MaxPool2d(2, 2),  # 16 x 32 x 32
            nn.Flatten(),
            nn.Linear(e1, e2),
            # nn.Linear(e2, e3),
        )
        self.decoder = nn.Sequential(
            # nn.Linear(e3, e2),
            nn.Linear(e2, e1),
            View([-1, 16, img_size//2, img_size//2]),
            # 8 x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2,
                               padding=0),  # 16 x 64 x 64
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


# In[]

device = 'cuda'
# model = Autoencoder().cpu()
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)


def get_lr():
    for param_group in optimizer.param_groups:
        return param_group['lr']


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
        train_loss += loss.item()*images.size(0)
    train_loss = train_loss/len(dataloader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Learning Rate', get_lr(), epoch)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, train_loss))

filename_pth = f'1.pth'
torch.save(model.state_dict(), os.path.join(weight_dir_path, filename_pth))


# In[]
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
# In[]
model_path = os.path.join(weight_dir_path, filename_pth)
model.load_state_dict(torch.load(model_path))
thres_list = []
result = dict()
for folder in ["kizu", "sabi", "sure"]:
    path = base + f"data/washer/washer_ng/{folder}"
    testloader, test_len = get_dataloader(
        path, mode='test', batch_size=4, shuffle=False, num_workers=8)
    dataiter = iter(testloader)
    # dataiter = iter(dataloader)
    # image_list =  np.array([], shape=(4, 3, 128, 128))
    # output_list =  np.array([], shape=(4, 3, 128, 128))
    all_labels = []
    for i in range(int(np.ceil(test_len/batch_size))):
        # for i in range(int((len(test_files)/batch_size))):
        images, labels = dataiter.next()
        images = images.to(device)
        # print(":::::::::::::::::::::::::::::::::::::::::::::::")
        # get sample outputs
        output = model(images)
        # prep images for display
        images = images.cpu().numpy()

        # print(":::::::::::::::::::::::::::::::::::::::::::::::")
        # output is resized into a batch of iages
        output = output.view(-1, 3, img_size, img_size)
        # use detach when it's an output that requires_grad
        output = output.detach().cpu().numpy()

        fscore = feature_score(images, output)
        tp = 100
        iscore = instance_score(fscore, tp)
        score = iscore

        # print(f'fscore:{fscore}')
        print(f'iscore:{iscore}')
        print(f'iscore range:{min(iscore)}, {max(iscore)}')
        print(f'threshold:{threshold}')
        labels = (score > threshold).astype(int)
        all_labels += list(labels)
        print(f'labels:{labels}')
        print(f'total:{sum(labels)}/{len(labels)}')
        print(f'percent:{round(sum(labels)/len(labels)*100, 3)}%')
    print(
        f"::::::::::::::::::::::{path.split('/')[-1]}:::::::::::::::::::::::::")
    result[folder] = round(sum(all_labels)/len(all_labels)*100, 3)
    print(f'total:{sum(all_labels)}/{len(all_labels)}')
    print(f'percent:{result[folder]}%')
    print(":::::::::::::::::::::::::::::::::::::::::::::::")
    # In[]

    ncols = 3
    fig, axes = plt.subplots(nrows=batch_size, ncols=ncols,
                            sharex=True, sharey=True, figsize=(2*ncols, 2*batch_size))

    axes[0, 0].set_title('image')
    axes[0, 1].set_title('output')
    axes[0, 2].set_title('heatmap')
    i = 0
    for img, ax in zip(images, axes[:, 0]):
        ax.imshow(tensor2np(img))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1
    for img, ax in zip(output, axes[:, 1]):
        ax.imshow(tensor2np(img))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1
    # for img, ax in zip(fscore, axes[:, i]):
    #     ax.imshow(tensor2np(img))
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     i += 1
    for img, ax in zip(fscore, axes[:, 2]):
        im = tensor2np(img)
        ax.imshow(np.dot(im[...,:3], [0.2989, 0.5870, 0.1140]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        i += 1
    # for img, ax in zip(fscore, axes[:, i]):
    #     ax.imshow(tensor2np(img)[:, :, 1])
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     i += 1
    # for img, ax in zip(fscore, axes[:, i]):
    #     ax.imshow(tensor2np(img)[:, :, 2])
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     i += 1
# """
print(f'result:{result}')

# %%
# a = np.empty((3,3))
# b = np.append([[0, 1, 2], [3, 4, 5]],[[6, 7, 8]], axis=0)
# c = np.append(a, b, axis=1)
# print(c)
# %%

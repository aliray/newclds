import os
import random
import time
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue,
    RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import timm
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import ImageFolder

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 500,
    'train_bs': 16,
    'val_bs': 32,
    'lr': 1e-4,
    'T_0': 10,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [43, 45, 46, 69],
    'weights': [1, 1, 1, 1],
    'expid': f'tf_efficientnet_b4_ns_{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'
}

root = '/home/gpuserver/dataset/'
train_csv_path = '/home/gpuserver/dataset/train.csv'
image_root = '/home/gpuserver/dataset/train_images'

train = pd.read_csv(train_csv_path)
train.head()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    im_bgr = cv2.imread(path, )
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def show_img_grid(path):
    plt.figure(figsize=(16, 8))
    data_groups = train.groupby(by='label')
    for k in data_groups.groups:
        plt.subplot(1, len(data_groups), k + 1)
        plt.title(f'label {k} size {len(data_groups.groups[k])}')
        imgpath = os.path.join(path, train.values[random.choice(data_groups.groups[k])][0])
        plt.imshow(get_img(imgpath))
    plt.show()


# show_img_grid(image_root)


class CassavaDataset(Dataset):
    def __init__(
            self, df, data_root, transforms=None, output_label=True
    ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']

        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])

        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img


def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Resize(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        # CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Resize(CFG['img_size'], CFG['img_size']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_2019_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize([CFG['img_size'], CFG['img_size']]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # for training only, need nightly build pytorch

    model_arch = CFG.get('model_arch')
    expid = CFG.get('expid')
    tb_path = f'./tb/{expid}/'
    chk_root_path = f'./chk/{expid}/'
    if not os.path.exists(chk_root_path):
        os.makedirs(chk_root_path)

    seed_everything(CFG['seed'])
    writer = SummaryWriter(tb_path)
    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(model_arch, train.label.nunique()).to(device)
    scaler = GradScaler()

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break

        best_acc = 0.0
        train_ = train.loc[trn_idx, :].reset_index(drop=True)
        train_ds = CassavaDataset(train_, image_root, transforms=get_train_transforms())
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms())
        train_2019_path = '/home/gpuserver/dataset/2019/train'
        train_2019 = ImageFolder(train_2019_path, transform=get_2019_transforms())
        train_ds = train_ds.__add__(train_2019)

        step_size_up = math.floor(len(train_ds) / CFG['train_bs']) * 4
        train_loader = DataLoader(
            train_ds,
            batch_size=CFG['train_bs'],
            num_workers=CFG['num_workers'],
            shuffle=True,
            pin_memory=False,
        )
        val_loader = DataLoader(
            valid_ds,
            batch_size=CFG['val_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        # 训练代码

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1,
                                                                    eta_min=CFG['min_lr'], last_epoch=-1)
        model.train()
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        dataset_sizes = {
            'train': len(train_ds),
            'val': len(valid_ds)
        }
        for epoch in range(CFG.get('epochs')):
            print('Epoch {}/{}'.format(epoch, CFG.get('epochs') - 1))
            print('-' * 100)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                for step, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with autocast():
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            if phase == 'train':
                                # loss.backward()
                                scaler.scale(loss).backward()
                                # optimizer.step()

                        # statistics
                        step_loss = loss.item() * inputs.size(0) / CFG.get(f'{phase}_bs')
                        step_acc = torch.sum(preds == labels.data) / CFG.get(f'{phase}_bs')
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        if phase == 'train':
                            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()

                        if step % CFG.get('verbose_step') == 0:
                            print(
                                f'fold {fold} {phase} step {step} running step lr '
                                f'{lr} '
                                f'step loss {step_loss} step acc {step_acc}')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    # exp_lr_scheduler.step()
                    writer.add_scalar(f'{phase} lr', lr, epoch)
                    for tag, parm in model.named_parameters():
                        writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)
                        # writer.add_histogram(tag + '_params', parm.data.cpu().numpy(), epoch)

                print(f'fold {fold} {phase} loss {epoch_loss} acc {epoch_acc} ')
                # deep copy the model
                if phase == 'val':
                    writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
                    writer.add_scalar(f'{phase} acc', epoch_acc, epoch)
                    torch.save(model.state_dict(), os.path.join(chk_root_path, f'{model_arch}_{fold}_{epoch}.pt'))

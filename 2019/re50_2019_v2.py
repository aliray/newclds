import os
import random
import time

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
    ShiftScaleRotate, CenterCrop, Resize, RandomShadow, RandomRotate90
)
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import math
from torchvision.datasets import ImageFolder
import timm
from tqdm.auto import tqdm, trange
from tempered_loss import bi_tempered_logistic_loss
import shutil


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_transforms(size):
    return Compose([
        RandomResizedCrop(size, size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms(size):
    return Compose([
        Resize(600, 600, p=1.),
        CenterCrop(size, size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_img(path):
    return cv2.imread(path)[:, :, ::-1]


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
        if self.output_label:
            return img, target
        else:
            return img


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, drop_rate=0, drop_path_rate=0, pretrained=True):
        super().__init__()
        self.drop_rate = drop_rate
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_class)

    def forward(self, x):
        x = self.model(x)
        x = nn.Dropout(self.drop_rate)(x)
        return x


def train(CFG):
    # for training only, need nightly build pytorch
    torch.cuda.empty_cache()
    time_stmp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    train_csv_path = '/home/gpuserver/dataset/2019/train_2019.csv'
    image_root = '/home/gpuserver/dataset/2019/train_2019_v2/train'
    train = pd.read_csv(train_csv_path)

    model_arch = CFG.get('model_arch')
    expid = CFG.get('expid')
    log_root = '/home/gpuserver/race/exps'
    tb_path = f'{log_root}/tb/{expid}/{time_stmp}'
    chk_root_path = f'{log_root}/chk/{expid}/{time_stmp}'

    if not os.path.exists(chk_root_path):
        os.makedirs(chk_root_path)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
        shutil.copy(__file__, os.path.join(tb_path, 'train.py'))

    seed_everything(CFG['seed'])
    writer = SummaryWriter(tb_path)
    device = torch.device(CFG['device'])

    folds = StratifiedKFold(n_splits=5).split(np.arange(train.shape[0]), train.label.values)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold not in CFG['folds']:
            continue

        model = CassvaImgClassifier(model_arch, 5,
                                    drop_rate=CFG['drop_rate']).to(
            device)
        scaler = GradScaler()

        train_ = train.loc[trn_idx, :].reset_index(drop=True)
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        train_ds = CassavaDataset(train_, image_root, transforms=get_train_transforms(CFG['img_size']))
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms(CFG['img_size']))

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
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        exp_lr_scheduler = \
            lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                     T_0=CFG['T_0'],
                                                     T_mult=1,
                                                     eta_min=CFG['min_lr'],
                                                     last_epoch=-1)
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        dataset_sizes = {
            'train': len(train_ds),
            'val': len(valid_ds)
        }

        besc_acc = 0.0
        rangebar = trange(CFG.get('epochs'))
        for epoch in rangebar:
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), desc=f'{epoch}')
                for step, (inputs, labels) in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with autocast():
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = bi_tempered_logistic_loss(outputs, labels, t1=CFG['t1'], t2=CFG['t2'],
                                                             label_smoothing=CFG['label_smoothing'])
                            if phase == 'train':
                                scaler.scale(loss).backward()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        if phase == 'train':
                            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()

                        if step == len(dataloaders[phase]):
                            pbar.set_description(desc=f'{epoch} {running_corrects.double() / dataset_sizes[phase]}')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase} acc', epoch_acc, epoch)
                if phase == 'train' and CFG['lrstep']:
                    exp_lr_scheduler.step()
                    writer.add_scalar(f'{phase} lr', lr, epoch)

                if phase == 'val':
                    if epoch_acc > besc_acc:
                        besc_acc = epoch_acc
                        rangebar.set_description(f'best {besc_acc:.3f}')
                        torch.save(model.state_dict(), os.path.join(chk_root_path, f'{model_arch}_best.pt'))

        del model


if __name__ == '__main__':
    confs = [
        {
            'fold_num': 5,
            'folds': [0],
            'seed': 719,
            'model_arch': 'resnet50',
            'img_size': 512,
            'epochs': 100,
            'train_bs': 16,
            'val_bs': 32,
            'lr': 1e-4,
            'T_0': 20,
            'min_lr': 1e-6,
            'weight_decay': 1e-6,
            'num_workers': 16,
            'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
            'verbose_step': 100,
            'drop_rate': 0.2,
            'device': 'cuda:0',
            'tta': 3,
            'used_epochs': [43, 45, 46, 69],
            'weights': [1, 1, 1, 1],
            't1': 1,
            't2': 1,
            'label_smoothing': 0.0,
            'expid': f'resnet50_2019_smooth',
            'lrstep': True,
        }
    ]
    print(confs)
    train(confs[0])

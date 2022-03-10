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
    ShiftScaleRotate, CenterCrop, Resize
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

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'densenet121',
    'img_size': 448,
    'epochs': 100,
    'train_bs': 16,
    'val_bs': 32,
    'lr': 1e-4,
    'T_0': 20,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 12,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [43, 45, 46, 69],
    'weights': [1, 1, 1, 1],
    'expid': f'densenet121_2019_{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_valid_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(CFG['img_size']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop([CFG['img_size'], CFG['img_size']]),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = torchvision.models.densenet121(pretrained=pretrained)
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
    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(model_arch, 5).to(device)
    scaler = GradScaler()
    train_path = '/home/gpuserver/dataset/2019/train'
    val_path = '/home/gpuserver/dataset/2019/val'
    train_ds = ImageFolder(train_path, transform=get_train_transforms())
    valid_ds = ImageFolder(val_path, transform=get_valid_transforms())

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

                # optimizer.zero_grad()
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            scaler.scale(loss).backward()

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
                        # writer.add_scalar(f'{phase} loss', step_loss, (epoch + 1) * (step + 1))

                    if step % CFG.get('verbose_step') == 0:
                        print(
                            f'{phase} step {step} running step lr '
                            f'{lr} '
                            f'step loss {step_loss} step acc {step_acc}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train' and epoch < 12:
                exp_lr_scheduler.step()
                writer.add_scalar(f'{phase} lr', lr, epoch)

            print(f' {phase} loss {epoch_loss} acc {epoch_acc} ')
            # deep copy the model
            if phase == 'val':
                writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase} acc', epoch_acc, epoch)
                torch.save(model.state_dict(), os.path.join(chk_root_path, f'{model_arch}_{epoch}.pt'))

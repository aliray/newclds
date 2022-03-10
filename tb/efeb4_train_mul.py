import os
import random
import time
import shutil
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
from torch.utils.tensorboard import SummaryWriter
import timm
from torch.cuda.amp import autocast, GradScaler
from loss.tempered_loss import bi_tempered_logistic_loss
from tqdm.auto import tqdm, trange
from loss.focal_cosine_loss import FocalCosineLoss
import math
from loss.rceloss import NFLandRCE

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 100,
    'train_bs': 32,
    'val_bs': 32,
    'lr': 1e-4,
    'T_0': 10,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    't1': 0.4,
    't2': 1.4,
    'num_workers': 12,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'folds': [0],
    'label_smoothing': 0.0,
    'drop_rate': 0.1,
    'num_classes': 5,
    'load2019': False,
    'path_2019': '/home/gpuserver/race/race_clds/2019/chk/efb4_2019_pre_0.928/tf_efficientnet_b4_ns_best.pt',
    'expid': f'e4_{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'
}

train_csv_path = '/root/datasets/train.csv'
image_root = '/root/datasets/train_images'
train = pd.read_csv(train_csv_path)


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
        target = None
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


def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=False,
                                       checkpoint_path='./tf_efficientnet_b4_ns-d6313a46.pth',
                                       drop_rate=CFG['drop_rate'])
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
        shutil.copy(__file__, os.path.join(chk_root_path, 'train.py'))

    seed_everything(CFG['seed'])
    writer = SummaryWriter(tb_path)
    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(CFG['device'])

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold not in CFG['folds']:
            continue

        model = CassvaImgClassifier(model_arch, CFG['num_classes']).to(device)
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

        if CFG['load2019']:
            model.load_state_dict(torch.load(CFG['path_2019']))

        scaler = GradScaler()
        train_ = train.loc[trn_idx, :].reset_index(drop=True)
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        train_ds = CassavaDataset(train_, image_root, transforms=get_train_transforms())
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms())

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

        criterion = bi_tempered_logistic_loss
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        exp_lr_scheduler = \
            lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                     T_0=CFG['T_0'],
                                                     T_mult=1,
                                                     eta_min=CFG['min_lr'],
                                                     last_epoch=-1)
        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_ds), 'val': len(valid_ds)}

        rangebar = trange(CFG.get('epochs'), desc=f'fold {fold}')
        besc_acc = 0.0
        for epoch in rangebar:
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0

                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), desc=f'{phase} {epoch}')
                for step, (inputs, labels) in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    with autocast():
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = bi_tempered_logistic_loss(outputs, labels,
                                                             t1=CFG['t1'], t2=CFG['t2'],
                                                             label_smoothing=CFG['label_smoothing'])
                            # NFLandRCE(alpha=1, beta=1, num_classes=5)(outputs, labels)
                            if phase == 'train':
                                scaler.scale(loss).backward()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        if phase == 'train':
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    exp_lr_scheduler.step()
                    writer.add_scalar(f'{phase} lr', lr, epoch)

                if phase == 'val':
                    torch.save(model.state_dict(), os.path.join(chk_root_path, f'{model_arch}_{fold}_{epoch}.pt'))
                    if epoch_acc > besc_acc:
                        besc_acc = epoch_acc
                        rangebar.set_description(f'fold {fold} best acc {besc_acc:5.3f}')

                writer.add_scalar(f'epoch {phase} loss', epoch_loss, epoch)
                writer.add_scalar(f'epoch {phase} acc', epoch_acc, epoch)

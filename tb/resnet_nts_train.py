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
    ShiftScaleRotate, CenterCrop, Resize,
    RandomGridShuffle, RandomRotate90
)
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import timm
from torch.cuda.amp import autocast, GradScaler
from tempered_loss import bi_tempered_logistic_loss
from loss.focal_cosine_loss import FocalCosineLoss
from loss.sce_loss import SCELoss
import math

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'resnet50',
    'img_size': 512,
    'reimg_size': 512,
    'epochs': 20,
    'train_bs': 16,
    'val_bs': 32,
    'lr': 1e-4,
    'T_0': 10,
    'min_lr': 1e-6,
    'weight_decay': 0,
    't1': 0.2,
    't2': 1.2,
    'alpha': 2.0,
    'beta': 1.0,
    'num_workers': 12,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'folds': [0],
    'label_smoothing': 0.55,
    'num_classes': 5,
    'load2019': False,
    'teaching': False,
    'teacherpt_path': '/home/gpuserver/race/race_clds/chk/'
                      'resnet50_pretrain_2021-01-22-23:48:52_0.898/resnet50_0_15.pt',
    'expid': f're50_nts_{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'
}

root = '/home/gpuserver/dataset/'
train_csv_path = '/home/gpuserver/dataset/train.csv'
image_root = '/home/gpuserver/dataset/train_images'
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
            self, df, data_root, transforms=None, output_label=True, teacher_target=None
    ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
        self.teacher_target = teacher_target

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        target = None
        if self.output_label:
            if self.teacher_target is not None:
                target = int(self.teacher_target[index])
            else:
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
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True, load2019=True):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


def get_pred(model, loader, output_softmax=True):
    preds = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc=f'loading tearcher')
        for data, _ in pbar:
            data = data.to(device)
            output = model(data)
            if output_softmax:
                pred = F.softmax(output, dim=1).cpu().numpy()
            else:
                pred = output.argmax(dim=1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)


def learning_rate(lr_init, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
    return lr_init * math.pow(0.2, optim_factor)


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
        if CFG['load2019']:
            model.load_state_dict(torch.load('/home/gpuserver/race/race_clds/2019/chk/'
                                             'resnet50_2019_2021-01-18-23:43:37/resnet50_77.pt'))

        scaler = GradScaler()
        train_ = train.loc[trn_idx, :].reset_index(drop=True)
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        train_ds = CassavaDataset(train_, image_root, transforms=get_train_transforms())
        train_loader = DataLoader(
            train_ds,
            batch_size=CFG['train_bs'],
            num_workers=CFG['num_workers'],
            shuffle=True,
            pin_memory=False,
        )
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms())
        val_loader = DataLoader(
            valid_ds,
            batch_size=CFG['val_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        tmodel = None
        if CFG['teaching']:
            train_ds = CassavaDataset(train_, image_root, transforms=get_valid_transforms())
            train_loader = DataLoader(
                train_ds,
                batch_size=CFG['train_bs'],
                num_workers=CFG['num_workers'],
                shuffle=True,
                pin_memory=False,
            )
            tmodel = CassvaImgClassifier(model_arch, CFG['num_classes']).to(device)
            tmodel.load_state_dict(torch.load(CFG['teacherpt_path']))
            preds = get_pred(tmodel, train_loader, output_softmax=False)
            del tmodel, train_ds, train_loader
            torch.cuda.empty_cache()
            train_ds = CassavaDataset(train_, image_root,
                                      transforms=get_train_transforms(), teacher_target=preds)
            train_loader = DataLoader(
                train_ds,
                batch_size=CFG['train_bs'],
                num_workers=CFG['num_workers'],
                shuffle=True,
                pin_memory=False,
            )

        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        criterion = bi_tempered_logistic_loss
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
                    print('target size', labels.size())
                    with autocast():
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = F.nll_loss(outputs, labels)

                            if phase == 'train':
                                scaler.scale(loss).backward()

                        # statistics
                        step_loss = loss.item() * inputs.size(0) / CFG.get(f'{phase}_bs')
                        step_acc = torch.sum(preds == labels.data) / CFG.get(f'{phase}_bs')
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        if phase == 'train':
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                    pbar.set_description(desc=f'{phase} {epoch} {step_acc:.3f}')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                if phase == 'train':
                    exp_lr_scheduler.step()
                    writer.add_scalar(f'{fold}-{phase}-lr', lr, epoch)

                if phase == 'val':
                    torch.save(model.state_dict(), os.path.join(chk_root_path, f'{model_arch}_{fold}_{epoch}.pt'))
                    if epoch_acc > besc_acc:
                        besc_acc = epoch_acc
                        rangebar.set_description(f'fold {fold} best acc {besc_acc:5.3f}')

                writer.add_scalar(f'{fold}-{phase} loss', epoch_loss, epoch)
                writer.add_scalar(f'{fold}-{phase} acc', epoch_acc, epoch)

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
    RandomGridShuffle
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
from tempered_loss import bi_tempered_logistic_loss
import math
from torchvision import transforms

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
    't1': 0.4,
    't2': 1.4,
    'num_workers': 12,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [43, 45, 46, 69],
    'folds': [0],
    'weights': [1, 1, 1, 1],
    'expid': f'ef4_lessaug+temp_{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'
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


def get_img_pil(path):
    from PIL import Image
    im = Image.open(path)
    return im

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

        img = get_img_pil(path)
        if self.transforms:
            img = self.transforms(img)

        # do label smoothing
        if self.output_label:
            return img, target
        else:
            return img


def scale_crop(size, num_crops=1):
    assert num_crops in [1, 5, 10], "num crops must be in {1,5,10}"
    convert_tensor = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if num_crops == 1:
        t_list = [
            transforms.CenterCrop(size),
            convert_tensor
        ]
    else:
        if num_crops == 5:
            t_list = [transforms.FiveCrop(size)]
        elif num_crops == 10:
            t_list = [transforms.TenCrop(size)]
        # returns a 4D tensor
        t_list.append(transforms.Lambda(lambda crops:
                                        torch.stack([convert_tensor(crop) for crop in crops])))

    return transforms.Compose(t_list)


def get_train_transforms():
    return scale_crop(CFG['img_size'], num_crops=5)


def get_valid_transforms():
    return transforms.Compose([
        transforms.CenterCrop(CFG['img_size']),
        transforms.Resize(CFG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
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
        shutil.copy(__file__, os.path.join(chk_root_path, 'train_0.918.py'))

    seed_everything(CFG['seed'])
    writer = SummaryWriter(tb_path)
    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(CFG['device'])

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold not in CFG['folds']:
            continue
        model = CassvaImgClassifier(model_arch, train.label.nunique()).to(device)
        scaler = GradScaler()
        best_acc = 0.0
        train_ = train.loc[trn_idx, :].reset_index(drop=True)
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        train_ds = CassavaDataset(train_, image_root, transforms=get_train_transforms())
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms())
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
        criterion = bi_tempered_logistic_loss
        optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        exp_lr_scheduler = \
            lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                     T_0=CFG['T_0'],
                                                     T_mult=1,
                                                     eta_min=CFG['min_lr'],
                                                     last_epoch=-1)
        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_ds), 'val': len(valid_ds)}
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
                            if phase == 'train':
                                bs, ncrops, c, h, w = inputs.size()
                                outputs = model(inputs.view(-1, c, h, w))
                                outputs = outputs.view(bs, ncrops, -1).mean(1)
                                _, preds = torch.max(outputs, 1)
                            else:
                                outputs = model(inputs)
                                _, preds = torch.max(outputs, 1)

                            loss = criterion(outputs, labels, t1=CFG['t1'], t2=CFG['t2'])
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

                        if step % CFG.get('verbose_step') == 0:
                            print(
                                f'fold {fold} {phase} step {step} running step lr '
                                f'{lr} loss {step_loss} step acc {step_acc}')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    exp_lr_scheduler.step()
                    writer.add_scalar(f'{phase} lr', lr, epoch)

                print(f'fold {fold} {phase} loss {epoch_loss} acc {epoch_acc} ')
                writer.add_scalar(f'epoch {phase} loss', epoch_loss, epoch)
                writer.add_scalar(f'epoch {phase} acc', epoch_acc, epoch)
                if phase == 'val':
                    torch.save(
                        model.state_dict(),
                        os.path.join(chk_root_path, f'{model_arch}_{fold}_{epoch}.pt')
                    )

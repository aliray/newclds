import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue,
    RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm

CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'resnet50',
    'img_size': 512,
    'epochs': 100,
    'train_bs': 16,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 12,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'tta': 5,
    'fold': [(0, [10, 11, 14, 15])],
    'weights': [1, 1, 1, 1],
    'expid': f'resnet50_pretrain_2021-01-22-23:48:52_0.898'

}

root = '/home/gpuserver/dataset/'
train_csv_path = '/home/gpuserver/dataset/train.csv'
image_root = '/home/gpuserver/dataset/train_images'


# root = 'F:/race/'
# train_csv_path = 'F:/race/train.csv'
# image_root = 'F:/race/train_images'


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
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


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


def get_inference_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)  # output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


if __name__ == '__main__':
    # for training only, need nightly build pytorch
    train = pd.read_csv(train_csv_path)
    model_arch = CFG.get('model_arch')
    expid = CFG.get('expid')
    tb_path = f'./tb/{expid}/'
    chk_root_path = f'./chk/{expid}/'

    seed_everything(CFG['seed'])
    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(CFG['device'])

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first

        for (used_fold, used_epochs) in CFG['fold']:
            if fold == used_fold:
                model = CassvaImgClassifier(model_arch, train.label.nunique()).to(device)
                model.eval()
                print('Inference fold {} started'.format(fold))
                test_path = os.path.join(root, 'test_images')
                test = pd.DataFrame()
                test['image_id'] = list(os.listdir(test_path))
                valid_ = train.loc[val_idx, :].reset_index(drop=True)
                valid_ds = CassavaDataset(valid_, image_root, transforms=get_inference_transforms(), output_label=False)
                val_loader = DataLoader(
                    valid_ds, batch_size=CFG['valid_bs'],
                    num_workers=CFG['num_workers'],
                    shuffle=False,
                    pin_memory=False,
                )

                test_ds = CassavaDataset(test, test_path, transforms=get_inference_transforms(), output_label=False)
                tst_loader = DataLoader(
                    test_ds,
                    batch_size=CFG['valid_bs'],
                    num_workers=CFG['num_workers'],
                    shuffle=False,
                    pin_memory=False,
                )

                val_preds = []
                tst_preds = []
                for i, epoch in enumerate(used_epochs):
                    model.load_state_dict(torch.load(os.path.join(chk_root_path, f'{model_arch}_{fold}_{epoch}.pt')))
                    with torch.no_grad():
                        for _ in range(CFG['tta']):
                            val_preds += [
                                CFG['weights'][i] / sum(CFG['weights']) / CFG['tta'] * inference_one_epoch(model,
                                                                                                           val_loader,
                                                                                                           device)
                            ]
                            # tst_preds += [
                            #     CFG['weights'][i] / sum(CFG['weights']) / CFG['tta'] * inference_one_epoch(model,
                            #                                                                                tst_loader,
                            #                                                                                device)]

                val_preds = np.mean(val_preds, axis=0)
                tst_preds = np.mean(tst_preds, axis=0)

                print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, val_preds)))
                print('fold {} validation accuracy = {:.5f}'.format(fold, (
                        valid_.label.values == np.argmax(val_preds, axis=1)).mean()))

                del model
                torch.cuda.empty_cache()

    # test['label'] = np.argmax(tst_preds, axis=1)
    # test.head()
    # test.to_csv('submission.csv', index=False)

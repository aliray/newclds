from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random

import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import cv2
import timm  # from efficientnet_pytorch import EfficientNet
from sklearn.metrics import log_loss
from pl_bolts.models.vision import UNet

CFG = {
    'fold_num': 10,
    'seed': 719,
    'model_arch': 'tf_efficientnet_b3_ns',
    'img_size': 224,
    'epochs': 100,
    'train_bs': 32,
    'valid_bs': 32,
    'lr': 1e-3,
    'num_workers': 4,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [16, 17, 18, 19],
    'weights': [1, 1, 1, 1],
    'is_test': True
}

root = '/home/gpuserver/dataset/'
# train_csv_path = '/home/gpuserver/dataset/train.csv'
# image_root = '/home/gpuserver/dataset/train_images'

train_csv_path = 'F:/race/train.csv'
image_root = 'F:/race/train_images'

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


show_img_grid(image_root)


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


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
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
        CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
        Resize(CFG['img_size'], CFG['img_size']),
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


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


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

    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        if fold > 0:
            break

        best_acc = 0.0
        train_ = train.loc[trn_idx, :].reset_index(drop=True)
        train_ds = CassavaDataset(train_, image_root, transforms=get_train_transforms())
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms())

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=CFG['train_bs'],
            num_workers=CFG['num_workers'],
            shuffle=True,
            pin_memory=False,
        )
        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        # ????????????
        import torch.optim as optim
        from torch.optim import lr_scheduler

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'])
        # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer, patience=3
        # )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.5)
        model.train()
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        dataset_sizes = {
            'train': len(train_ds),
            'val': len(valid_ds)
        }

        if not CFG.get('is_test', False):
            for epoch in range(CFG.get('epochs')):
                print('Epoch {}/{}'.format(epoch, CFG.get('epochs') - 1))
                print('-' * 100)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for step, (inputs, labels) in enumerate(dataloaders[phase]):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        step_loss = loss.item() * inputs.size(0) / CFG.get('train_bs')
                        step_acc = torch.sum(preds == labels.data) / CFG.get('train_bs')
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        if step % CFG.get('verbose_step') == 0:
                            print(
                                f'fold {fold} {phase} step {step} running step lr '
                                f'{optimizer.state_dict()["param_groups"][0]["lr"]} '
                                f'step loss {step_loss} step acc {step_acc}')

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    if phase == 'train':
                        exp_lr_scheduler.step()

                    print(f'fold {fold} {phase} loss {epoch_loss} acc {epoch_acc} ')
                    # deep copy the model
                    if phase == 'val':
                        torch.save(model.state_dict(),
                                   f'./chk/{CFG.get("model_arch")}_{fold}_{epoch}.pt')
                        if epoch_acc > best_acc:
                            best_acc = epoch_acc
                            torch.save(model.state_dict(),
                                       f'./chk/{CFG.get("model_arch")}_best_model_{fold}.pt')

                # print()
                """ >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ????????????????????? """

        else:

            print('Inference fold {} started'.format(fold))
            test_path = os.path.join(root, 'test_images')
            test = pd.DataFrame()
            test['image_id'] = list(os.listdir(test_path))

            valid_ds = CassavaDataset(valid_, image_root, transforms=get_inference_transforms(), output_label=False)
            val_loader = torch.utils.data.DataLoader(
                valid_ds,
                batch_size=CFG['valid_bs'],
                num_workers=CFG['num_workers'],
                shuffle=False,
                pin_memory=False,
            )

            test_ds = CassavaDataset(test, test_path, transforms=get_inference_transforms(), output_label=False)
            tst_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=CFG['valid_bs'],
                num_workers=CFG['num_workers'],
                shuffle=False,
                pin_memory=False,
            )

            val_preds = []
            tst_preds = []
            for i, epoch in enumerate(CFG['used_epochs']):
                model.load_state_dict(torch.load(f'./chk/{CFG.get("model_arch")}_{fold}_{epoch}.pt'))
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

    # if len(tst_preds) > 0:
    #     test['label'] = np.argmax(tst_preds, axis=1)
    #     test.head()
    #     test.to_csv('submission.csv', index=False)

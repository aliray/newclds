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
from tempered_loss import bi_tempered_logistic_loss
from loss.focal_cosine_loss import FocalCosineLoss
CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'resnet_consine_tempered_loss',
    'img_size': 512,
    'epochs': 100,
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
    'expid': f'resnet_consine_tempered_loss_{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'
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

        if self.output_label == True:
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
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_class)

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


classes = [0, 1, 2, 3, 4]


def matplotlib_imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)


def images_to_probs(net, images):
    import torch.nn.functional as F
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


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
        import math

        step_size_up = math.floor(len(train_ds) / CFG['train_bs']) * 2
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

        criterion = FocalCosineLoss()
        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
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

                    # optimizer.zero_grad()
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
                            # writer.add_scalar(f'{phase} loss', step_loss, (epoch + 1) * (step + 1))

                        if step % CFG.get('verbose_step') == 0:
                            print(
                                f'fold {fold} {phase} step {step} running step lr '
                                f'{lr} '
                                f'step loss {step_loss} step acc {step_acc}')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    exp_lr_scheduler.step()
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

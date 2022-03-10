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
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 100,
    'train_bs': 16,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 10,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [6, 7, 8, 9],
    'weights': [1, 1, 1, 1],
    'expid': f'tf_efficientnet_b4_ns_2020-12-15-19:00:06'

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
        if self.output_label == True:
            return img, target
        else:
            return img


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, n_class)

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
        # RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        # ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        # Cutout(p=0.5),
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
        # Resize(CFG['img_size'], CFG['img_size']),
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


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


classes = ["0",
           "1",
           "2",
           "3",
           "4"]


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(16, 12))
    for idx in np.arange(CFG['valid_bs']):
        ax = fig.add_subplot(4, 8, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red")
        )
    return fig, preds


def inference_one_epoch(e, model, data_loader, device, writer):
    model.eval()

    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        image_preds = model(imgs)  # output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        writer.add_figure(f'iamges - {e} - {step}', plot_classes_preds(model, imgs, labels), global_step=step)
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def ana_one_epoch(e, model, data_loader, device, writer):
    model.eval()
    image_preds_all, labels_all = [], []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        if len(imgs) != CFG['valid_bs']:
            continue
        fig, preds = plot_classes_preds(model, imgs, labels)
        image_preds_all += [preds]
        labels_all += labels
        writer.add_figure(f'iamges - {e} - {step}', fig, global_step=step)
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    # labels_all = np.concatenate(labels_all, axis=0)
    from sklearn.metrics import classification_report
    print(f'model {e} report')
    print(classification_report(labels_all, image_preds_all, target_names=classes))


if __name__ == '__main__':
    # for training only, need nightly build pytorch
    train = pd.read_csv(train_csv_path)
    model_arch = CFG.get('model_arch')
    expid = CFG.get('expid')
    tb_path = f'./tb/{expid}/'
    chk_root_path = f'./chk/{expid}/'
    writer = SummaryWriter(tb_path)
    seed_everything(CFG['seed'])
    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(model_arch, train.label.nunique()).to(device)
    model.eval()

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        if fold > 0:
            break

        print('Inference fold {} started'.format(fold))
        test_path = os.path.join(root, 'test_images')
        test = pd.DataFrame()
        test['image_id'] = list(os.listdir(test_path))
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms(), output_label=True)
        val_loader = DataLoader(
            valid_ds, batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        for i, epoch in enumerate(CFG['used_epochs']):
            model.load_state_dict(torch.load(os.path.join(chk_root_path, f'{model_arch}_{fold}_{epoch}.pt')))
            with torch.no_grad():
                ana_one_epoch(epoch,
                              model,
                              val_loader,
                              device,
                              writer)

        del model
        torch.cuda.empty_cache()

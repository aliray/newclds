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
import timm
import argparse
from torch.cuda.amp import autocast, GradScaler
from loss.tempered_loss import bi_tempered_logistic_loss
from utils import seed_everything
from datasets import CassavaDataset, get_train_transforms, get_valid_transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from loss.sce_loss import SCELoss

MODEL = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'densenet121': torchvision.models.densenet121
}
LOSS = {}
OPTIM = {}


def get_optimizer(config):
    return None


def get_loss(config):
    return bi_tempered_logistic_loss


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = MODEL.get(model_arch)(pretrained=pretrained)
        if model_arch == 'densenet121':
            self.model.classifier = nn.Linear(self.model.classifier.in_features, n_class)
        else:
            self.model.fc = nn.Linear(self.model.fc.in_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


def run_trainepoch(epoch, scaler, model, optimizer, criterion, dataloader, size, device, config, args):
    model.train()
    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with autocast():
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels, t1=config['t1'], t2=config['t2'],
                                 label_smoothing=config['label_smoothing'])
                scaler.scale(loss).backward()
            if ((step + 1) % args['accum_iter'] == 0) or ((step + 1) == len(dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


def run_valepoch(epoch, model, optimizer, criterion, dataloader, size, device, config, args):
    model.eval()
    running_corrects = 0
    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with autocast():
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / size
    tune.report(accuracy=epoch_acc.item())


def main(config, args, checkpoint_dir=None):
    seed_everything(args['seed'])
    root = args['root']
    image_root = os.path.join(root, 'train_images')
    pdf = pd.read_csv(os.path.join(root, 'train.csv'))
    folds = StratifiedKFold(n_splits=args['fold_num']).split(np.arange(pdf.shape[0]), pdf.label.values)
    device = torch.device('cuda:0')

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break
        model = CassvaImgClassifier(config['model_arch'], pdf.label.nunique()).to(device)
        scaler = GradScaler()
        train_ds = CassavaDataset(
            pdf.loc[trn_idx, :].reset_index(drop=True),
            image_root,
            transforms=get_train_transforms(config['image_size'])
        )
        valid_ds = CassavaDataset(
            pdf.loc[val_idx, :].reset_index(drop=True),
            image_root,
            transforms=get_valid_transforms(config['image_size'])
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            num_workers=args['num_workers'],
            shuffle=True,
            pin_memory=False,
        )
        val_loader = DataLoader(
            valid_ds,
            batch_size=config['batch_size'],
            num_workers=args['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        criterion = bi_tempered_logistic_loss
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
        exp_lr_scheduler = \
            lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                     T_0=config['T_0'],
                                                     T_mult=config['T_mult'],
                                                     eta_min=config['min_lr'],
                                                     last_epoch=-1)
        for epoch in range(1):
            run_trainepoch(
                epoch,
                scaler,
                model,
                optimizer,
                criterion,
                dataloader=train_loader,
                size=len(train_ds),
                device=device,
                config=config,
                args=args
            )
            run_valepoch(
                epoch,
                model,
                optimizer,
                criterion,
                dataloader=val_loader,
                size=len(valid_ds),
                device=device,
                config=config,
                args=args
            )
            exp_lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", default=719, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--model_arch", default='resnet50')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--verbose_step", default=10, type=int)
    parser.add_argument("--tta", default=3, type=int)
    parser.add_argument("--root", default='/home/gpuserver/dataset/')
    parser.add_argument("--expid", default=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

    args, _ = parser.parse_known_args()

    config = {
        "T_0": 10,
        "T_mult": 1,
        "model_arch": "densenet121",
        "image_size": 512,
        "t1": 0.3,
        "t2": tune.quniform(1.1, 2, 0.1),
        "lr": 1e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.0,
        "batch_size": 16,
        "label_smoothing": 0.3,
    }
    from ray.tune.suggest.hyperopt import HyperOptSearch

    hyperopt_search = HyperOptSearch(
        metric="accuracy",
        mode="max",
        # points_to_evaluate=[
        #     {
        #         "T_0": 10,
        #         "T_mult": 1,
        #         "model_arch": "resnet50",
        #         "image_size": 512,
        #         "t1": 0.2,
        #         "t2": 1.2,
        #         "lr": 1e-4,
        #         "min_lr": 1e-6,
        #         "weight_decay": 0.0,
        #         "batch_size": 16,
        #         "label_smoothing": 0.5
        #     }
        # ],
    )
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max")
    result = tune.run(
        tune.with_parameters(main, args=args.__dict__),
        name=args.__dict__['name'],
        resources_per_trial={"cpu": 6, "gpu": 1},
        config=config,
        # metric="accuracy",
        # mode="max",
        num_samples=300,
        scheduler=scheduler,
        search_alg=hyperopt_search,
        local_dir='../raysearch',
    )
    best_trial = result.get_best_trial("accuracy", "max", "all")
    print("Best trial config: {}".format(best_trial.config))
    # checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    # print(f"best model path {checkpoint_path}")

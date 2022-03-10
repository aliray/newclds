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

MODEL = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
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
                # _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels, t1=config['t1'], t2=config['t2'])
                scaler.scale(loss).backward()

            if ((step + 1) % args['accum_iter'] == 0) or ((step + 1) == len(dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


def run_valepoch(epoch, model, optimizer, criterion, dataloader, size, device, config, args):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with autocast():
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels, t1=config['t1'], t2=config['t2'])

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size

    # with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
    #     path = os.path.join(checkpoint_dir, f"checkpoint-{epoch}.pt")
    #     torch.save(model.state_dict(), path)
    tune.report(loss=epoch_loss, accuracy=epoch_acc.item())


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
        # optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        # exp_lr_scheduler = \
        #     lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer,
        #         T_0=config['T_0'],
        #         T_mult=1,
        #         eta_min=config['min_lr'],
        #         last_epoch=-1
        #     )
        import math
        step_size_up = math.floor(len(train_ds) / config['batch_size']) * 2
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            momentum=config['momentum']
        )
        exp_lr_scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config['min_lr'],
            max_lr=config['lr'],
            step_size_up=step_size_up,
            base_momentum=config['momentum'],
            max_momentum=0.9,
        )
        for epoch in range(1):
            exp_lr_scheduler.step()
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
            # exp_lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", default=719, type=int)
    parser.add_argument("--fold_num", default=5, type=int)
    parser.add_argument("--model_arch", default='tf_efficientnet_b4_ns')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--verbose_step", default=10, type=int)
    parser.add_argument("--tta", default=3, type=int)
    parser.add_argument("--root", default='/home/gpuserver/dataset/')
    parser.add_argument("--expid", default=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

    args, _ = parser.parse_known_args()

    config = {
        "model_arch": tune.choice(["resnet50"]),
        "image_size": tune.choice([448]),
        # "T_0": tune.choice([8, 10, 12, 14]),
        # "T_MULT": tune.choice([0.5, 1, 1.5]),
        "t1": tune.quniform(0.1, 1, 0.1),
        "t2": tune.quniform(1, 4, 0.1),
        "lr": tune.choice([1e-2, 1e-3, 1e-4]),
        "min_lr": tune.choice([1e-5, 1e-6, 1e-7, 1e-8]),
        "weight_decay": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "momentum": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "batch_size": tune.choice([32])
    }
    from ray.tune.suggest.hyperopt import HyperOptSearch

    hyperopt_search = HyperOptSearch(metric="accuracy", mode="max")
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max")
    result = tune.run(
        tune.with_parameters(main, args=args.__dict__),
        name=args.__dict__['name'],
        resources_per_trial={"cpu": 12, "gpu": 1},
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

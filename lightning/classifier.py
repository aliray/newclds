import os
import torch
from torchvision import transforms, models, datasets
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
import pandas as pd
from utils import seed_everything, get_train_transforms, get_valid_transforms
import time
from sklearn.model_selection import StratifiedKFold
from dataset import CassavaDataset
from torch.utils.data import Dataset, DataLoader
import math
import torchvision


class ImageClassifier(pl.LightningModule):

    def __init__(self, model, loss=nn.CrossEntropyLoss(), optimizers=Adam):
        super(ImageClassifier, self).__init__()
        self.model = model
        self.loss = loss
        self.accuracy = pl.metrics.Accuracy()
        self.optimizers = optimizers
        self.lr = 1e-4

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return {'val_loss': loss, 'val_acc': self.accuracy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'val_loss': val_loss, 'val_acc': val_acc}
        return {'val_loss': val_loss, 'val_acc': val_acc, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizers(self.model.parameters(), lr=self.lr),
        }


if __name__ == '__main__':
    CFG = {
        'fold_num': 5,
        'seed': 719,
        'model_arch': 'tf_efficientnet_b3_ns',
        'img_size': 224,
        'epochs': 500,
        'train_bs': 8,
        'val_bs': 32,
        'lr': 1e-3,
        'T_0': 10,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'accum_iter': 1,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
        'verbose_step': 10,
        'device': 'cuda:0',
        'tta': 3,
        'used_epochs': [43, 45, 46, 69],
        'weights': [1, 1, 1, 1],
        'expid': f'tf_efficientnet_b3_ns_cosine{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}'
    }

    root = '/home/gpuserver/dataset/'
    train_csv_path = '/home/gpuserver/dataset/train.csv'
    image_root = '/home/gpuserver/dataset/train_images'
    train = pd.read_csv(train_csv_path)
    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    n_class = 5
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break

        best_acc = 0.0
        train_ = train.loc[trn_idx, :].reset_index(drop=True)
        valid_ = train.loc[val_idx, :].reset_index(drop=True)
        train_ds = CassavaDataset(train_, image_root, transforms=get_train_transforms(CFG['img_size']))
        valid_ds = CassavaDataset(valid_, image_root, transforms=get_valid_transforms(CFG['img_size']))

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
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_class)
        classifier = ImageClassifier(model=model)
        trainer = pl.Trainer(
            gpus=1,
            progress_bar_refresh_rate=20,
            val_check_interval=1,
            min_epochs=100,
            max_epochs=1000,
        )
        lr_finder = trainer.tuner.lr_find(classifier, train_loader, val_loader, num_training=1000)
        # trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=val_loader)
        fig = lr_finder.plot()
        fig.show()
        suggested_lr = lr_finder.suggestion()
        print(suggested_lr)


import os
import torch
from torchvision import transforms, models, datasets
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl


class ImageClassifier(pl.LightningModule):

    def __init__(self, args, config):
        super(ImageClassifier, self).__init__()
        # BaseClassifier.__init__(self, args, job)
        self.args = dict(args)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(args['class_names']))

        #
        self.best_val_loss = 1.0
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.accuracy = pl.metrics.Accuracy()
        #

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return {'val_loss': loss, 'val_acc': self.accuracy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.best_val_loss = avg_loss if self.best_val_loss > avg_loss else self.best_val_loss
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        # y, y_hat = y.to(self.__device), y_hat.to(self.__device)
        return {'test_loss': nn.CrossEntropyLoss()(y_hat, y), 'test_acc': self.accuracy(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss, 'avg_acc': avg_acc}
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), lr=1e-4)
        # optimizer_ft = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return {
            "optimizer": opt,
            # "lr_scheduler": lr_scheduler.ReduceLROnPlateau(opt, patience=3),
            # "monitor": "val_loss"
        }
        # return [opt], [lr_scheduler.ReduceLROnPlateau(opt, patience=3)]
        # torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

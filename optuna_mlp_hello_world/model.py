import os

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .config import Settings

settings = Settings()


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = build_mlp_network()
        self.ce = nn.CrossEntropyLoss()

    def build_mlp_network(no_of_layers, no_hidden_units):

        layers = []

        for i in range(no_of_layers):
            out_features = trial.suggest_int()

            layers.append(nn.Linear(64, 32))
            layers.append(nn.ReLU())

            dropout_prob = trial.suggest_float()
            layers.append(nn.Dropout(p))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    pl.seed_everything(42)
    mlp = MLP()
    trainer = pl.Trainer(
        auto_scale_batch_size="power", gpus=0, deterministic=True, max_epochs=5
    )
    trainer.fit(mlp, DataLoader(dataset))

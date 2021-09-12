import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from .config import Settings

settings = Settings()


class MLP(pl.LightningModule):
    def __init__(self, trial):
        super().__init__()

        # Initialise Optuna hyperparameter search space
        self.trial = trial
        self.lr = self.trial.suggest_loguniform(
            "learning rate",
            settings.LEARNING_RATE_MIN,
            settings.LEARNING_RATE_MAX,
        )
        self.dropout_prob = self.trial.suggest_float(
            "dropout_prob",
            settings.DROPOUT_RATE_MIN,
            settings.DROPOUT_RATE_MAX,
        )
        self.no_of_layers = self.trial.suggest_int(
            "no_of_layers",
            settings.NO_OF_LAYERS_MIN,
            settings.NO_OF_LAYERS_MAX,
        )

        # Preprocessing to be used by lightning data loaders
        self.transform = self.preprocess_mnist()

        # Build the network
        self.loss = nn.CrossEntropyLoss()
        self.layers = self.build_mlp_network()

    def build_mlp_network(self):
        layers = []

        input_len = settings.INPUT_LEN

        for i in range(self.no_of_layers):
            no_of_units = self.trial.suggest_int(
                f"no_of_units_l{i}",
                settings.HIDDEN_UNITS_MIN,
                settings.HIDDEN_UNITS_MAX,
            )

            layers.append(nn.Linear(input_len, no_of_units))
            layers.append(nn.ReLU())

            layers.append(nn.Dropout(self.dropout_prob))

            input_len = no_of_units  # Set input size of next layer

        # Make a prediction
        layers.append(nn.Linear(input_len, settings.NO_OF_CLASSES))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size()[0], -1)  # Unrow image
        y_hat = self.layers(x)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        data, target = batch
        data = data.reshape(data.size()[0], -1)

        output = self.layers(data)

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def preprocess_mnist() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # Download only
        MNIST(
            settings.DATA_DIR,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        MNIST(
            settings.DATA_DIR,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

    def setup(self, stage=None):
        # Transform the dataset
        mnist_train = MNIST(
            settings.DATA_DIR,
            train=True,
            download=False,
            transform=self.transform,
        )
        mnist_test = MNIST(
            settings.DATA_DIR,
            train=False,
            download=False,
            transform=self.transform,
        )

        # Train/val split
        mnist_train, mnist_val = random_split(mnist_train, [50_000, 10_000])

        small_data_subset = True
        if small_data_subset:
            mnist_train, _ = random_split(mnist_train, [500, 49_500])
            mnist_val, _ = random_split(mnist_val, [100, 9_900])
            mnist_test, _ = random_split(mnist_test, [100, 9_900])

        # Assign to use in DataLoaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=settings.BATCH_SIZE,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=settings.BATCH_SIZE,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=settings.BATCH_SIZE,
        )

import optuna
import pytorch_lightning as pl

from .config import Settings
from .model import MLP

settings = Settings()


def fit_trial(trial):
    pl.seed_everything(42)

    mlp = MLP(trial)

    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(mlp)


def main():
    study = optuna.create_study()
    study.optimize(fit_trial, n_trials=10)

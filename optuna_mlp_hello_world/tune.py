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

    return trainer.callback_metrics["val_acc"].item()


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(fit_trial, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

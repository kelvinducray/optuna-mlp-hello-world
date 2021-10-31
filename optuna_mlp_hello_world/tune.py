import logging

import optuna
import pytorch_lightning as pl
from optuna.trial import Trial

from .config import Settings
from .model import MLP

settings = Settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_trial(trial: Trial) -> float:
    pl.seed_everything(42)

    mlp = MLP(trial)

    trainer = pl.Trainer(max_epochs=settings.MAX_EPOCHS_PER_TRIAL)
    trainer.fit(mlp)

    return trainer.callback_metrics["val_acc"].item()


def main() -> None:
    logger.info("Starting hyper-parameter tuning...")
    study = optuna.create_study(
        study_name="MNIST Classifier",
        direction="maximize",
    )
    study.optimize(fit_trial, n_trials=settings.NO_OF_TRIALS)

    logger.info("Number of finished completed: %s", len(study.trials))

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info("  - Value: %s", trial.value)

    logger.info("  - Parameters: ")
    for key, value in trial.params.items():
        logger.info("    - %s: %s", key, value)

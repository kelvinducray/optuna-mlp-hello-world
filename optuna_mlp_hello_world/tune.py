import optuna

from .config import Settings

settings = Settings()


def objective(trial):

    ...

    return eval_metric


def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)

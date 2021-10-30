from pydantic import BaseSettings


class Settings(BaseSettings):
    # General training settings
    DATA_DIR: str = "./data"
    BATCH_SIZE: int = 64
    INPUT_LEN: int = 28 ** 2  # Unrowed MNIST image
    NO_OF_CLASSES: int = 10

    # Optuna hyperparameter search space
    NO_OF_LAYERS_MIN: int = 1
    NO_OF_LAYERS_MAX: int = 10

    HIDDEN_UNITS_MIN: int = 1
    HIDDEN_UNITS_MAX: int = 100

    LEARNING_RATE_MIN: float = 1e-5
    LEARNING_RATE_MAX: float = 0.2

    DROPOUT_RATE_MIN: int = 0
    DROPOUT_RATE_MAX: int = 1

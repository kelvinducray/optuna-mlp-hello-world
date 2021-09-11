class Settings:
    # General training settings
    DATA_DIR = "./data"
    BATCH_SIZE = 64
    INPUT_LEN = 28 ** 2  # Unrowed MNIST image
    NO_OF_CLASSES = 10

    # Optuna hyperparameter search space
    NO_OF_LAYERS_MIN = 1
    NO_OF_LAYERS_MAX = 10

    HIDDEN_UNITS_MIN = 1
    HIDDEN_UNITS_MAX = 100

    LEARNING_RATE_MIN = 1e-5
    LEARNING_RATE_MAX = 0.2

    DROPOUT_RATE_MIN = 0
    DROPOUT_RATE_MAX = 1

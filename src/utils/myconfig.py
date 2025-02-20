import sys
import os

sys.path.append("src")
import json
from dataclasses import asdict, dataclass, field

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

CONFIG_PATH = r"src\config.json"

@dataclass
class SaveConfig:
    # --------------Folders/Paths----------------------------------------
    SAVE_FOLDER: str = rf"data\final\Test"
    ENSEMBLE_LOAD_PATH: str = (
        rf"data\final\Run_3_FERM\Ensemble\All_data_metrics_ensemble.csv"
    )
    # -------------LSTM Model--------------------------------------------
    N_STEPS: int = 23
    # -------------General Model-----------------------------------------
    EARLY_STOP_PATIENCE: int = 50
    EARLY_STOP_MONITOR: str = "val_loss"
    EARLY_STOP_MODE: str = "min"
    ADAM_LEARN_SPEED: float = 0.001
    ALL_COLUMNS: list = field(default_factory=list)
    CORES = 4
    # ------------Data Preparation---------------------------------------
    K_FOLD_SPLITS: int = 5
    LOAD: str = "FERM"
    LABEL: str = "Protein"
    COMB_MIN_LENGTH: int = 3
    FILTER_QUANTILE: float = 0.10  # 10%

    # -----------Modes---------------------------------------------------
    USE_COMBINATIONS: bool = True
    RETRAIN: bool = False
    NUM_OF_IMG_TO_SAVE: int = 5
    STOP_AFTER_IMG_SAVE: bool = False
    SAVE_MODELS: bool = True
    COMMITEE_SEARCH: bool = True


@dataclass
class ModelConfig(SaveConfig):
    def __post_init__(self):
        # --------------Folders/Paths----------------------------------------
        self.RETRAIN_SAVE_PATH = rf"{self.SAVE_FOLDER}\Retrained"
        self.SHEETS_PATH = rf"{self.RETRAIN_SAVE_PATH}\sheets"
        self.MODEL_SAVE_PATH = rf"{self.RETRAIN_SAVE_PATH}\Models"

        self.MAIN_SAVE_METRICS_CSV = rf"{self.SAVE_FOLDER}\All_data_metrics_main.csv"
        self.RETRAIN_SAVE_METRICS_CSV = (
            rf"{self.RETRAIN_SAVE_PATH}\All_data_metrics_retrain.csv"
        )
        self.RETRAIN_METRICS_CSV = rf"{self.SAVE_FOLDER}\Filtered_data.csv"

        self.ENSEMBLE_SAVE_PATH = rf"{self.SAVE_FOLDER}\Ensemble"

        # -------------General Model-----------------------------------------
        self.callback: tf.keras.callbacks.EarlyStopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor=self.EARLY_STOP_MONITOR,
                mode=self.EARLY_STOP_MODE,
                patience=self.EARLY_STOP_PATIENCE,
            )
        )
        self.custom_optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(
            self.ADAM_LEARN_SPEED
        )

        # ------------Data Preparation---------------------------------------

        self.scaler = MinMaxScaler()
        self.kfold = KFold(n_splits=self.K_FOLD_SPLITS)


def load_config():
    """
    Reads the configuration from CONFIG_PATH and returns the JSON data.
    """
    with open(CONFIG_PATH, "r") as config_file:
        return json.load(config_file)


def save_config_json(save_config):
    """
    Saves the configuration to CONFIG_PATH using the dataclass representation.
    """
    with open(CONFIG_PATH, "w") as config_file:
        json.dump(asdict(save_config), config_file, indent=4)


def get_myconfig():
    """
    Loads the configuration from the JSON file and returns a ModelConfig instance.
    This function should be called in the main process only.
    """
    config_data = load_config()
    # Create the basic config object
    save_config = SaveConfig(**config_data)
    # Return the extended configuration with additional computed paths and objects
    return ModelConfig(**asdict(save_config))

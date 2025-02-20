import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import subprocess
from dataclasses import asdict

from utils.myconfig import get_myconfig, save_config_json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(CONFIG_PATH, "r") as config_file:
    content = config_file.read()
    print("Config file content:", content)  # Debug print
    config_data = json.loads(content)

def load_config():
    try:
        with open(CONFIG_PATH, "r") as config_file:
            return json.load(config_file)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON from {CONFIG_PATH}: {e}")
        raise


def save_config_json(save_config):
    with open(CONFIG_PATH, "w") as config_file:
        json.dump(asdict(save_config), config_file, indent=4)


def main():
    python_path = rf".venv\Scripts\python.exe"

    # Load the configuration once in the main process
    myconfig = get_myconfig()

    # Modify the configuration
    myconfig.ALL_COLUMNS = [
        "Time",
        "Age",
        "BiomassEst",
        "dBiomassEst",
        "BioWeight",
        "CPR",
        "Carbon_Feed",
        "dCarbon_Feed",
        "Biomass0",
        "Induction",
        "CumulativeAge",
        "miuSimp",
        "Protein",
    ]
    myconfig.SAVE_FOLDER = rf"data\final\Run_4_FERM"
    myconfig.K_FOLD_SPLITS = 5  # GSK - 3, FERM - 5
    myconfig.LOAD = "FERM"
    myconfig.RETRAIN = False
    myconfig.CORES = 4

    # Save the modified configuration
    save_config_json(myconfig)

    # Run subprocesses
    train_state = subprocess.run([python_path, r"src\training\train_model.py"], shell=True)
    train_state = True
    if train_state:
        filtering_state = subprocess.run(
            [python_path, r"src\utils\result_filtering.py"], shell=True)
    else:
        print("Failed to train")

    # Load the configuration again
    myconfig = get_myconfig()

    # Modify and save again
    myconfig.RETRAIN = True
    save_config_json(myconfig)

    # Run additional subprocesses
    filtering_state = True
    if filtering_state:
        retrain_state = subprocess.run(
            [python_path, r"src\training\train_model.py"], shell=True
        )
    else:
        print("Failed to filter")
    if retrain_state:
        ensemble_state = subprocess.run(
            [python_path, r"src\training\ensemble_search.py"], shell=True
        )
    else:
        print("Failed to retrain")
    if ensemble_state:
        print("Research done")
    else:
        print("Failed to build ensemble")


if __name__ == "__main__":
    main()

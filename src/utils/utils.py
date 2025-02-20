import csv
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from utils.myconfig import myconfig

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.options.mode.chained_assignment = None


def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.float32):
        return float(obj)  # Convert np.float32 to float
    return obj


def saveModelStructureInit(location):
    filename = rf"{location}\ModelStructure.txt"
    with open(filename, mode="w", newline="") as file:
        file.write("Model Structure \n\n")


def saveModelStructure_Model(location, model, optimizer):
    filename = rf"{location}\ModelStructure.txt"

    with open(filename, mode="w", newline="") as file:
        file.write(f"Loss Function: {model.loss} \n")
        # file.write(f'Optimizer: {model.optimizer} \n')
        file.write(f"Metrics: {model.metrics_names} \n")

        file.write(f"Optimizer: \n")
        json.dump(
            json.loads(
                json.dumps(optimizer.get_config(), default=convert_to_json_serializable)
            ),
            file,
            indent=4,
        )
        file.write("\n\n")

        for layer in model.layers:
            file.write(f"Layer Type: {type(layer).__name__} \n")
            json.dump(layer.get_config(), file, indent=4)
            file.write("\n")


def saveMeanMetrics_Init(file_name):
    # Create or overwrite the CSV file and write the headers
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "Test_ID",
            "Inputs",
            "MMSE",
            "MAE",
            "MSE",
            "RMSE",
            "RSS",
            "R2",
            "Model_id",
            "Weights",
        ]
        writer.writerow(headers)


def saveMeanMetrics(file_name, row):
    # Reopen the file in append mode and return the writer
    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def LoadTrainedInputs(csv_file) -> tuple:
    if os.path.exists(csv_file):
        savedData = pd.read_csv(f"{csv_file}")
        if savedData.shape[0] == 0:
            pass
        savedInputs = savedData["Inputs"].apply(eval).tolist()
    else:
        saveMeanMetrics_Init(csv_file)
        savedInputs = []

    return savedInputs


def load_inputs_csv(path, sort: bool = False):
    if os.path.exists(path):
        data = pd.read_csv(path)
        if sort:
            data.sort_values("MAE", axis=0, inplace=True)
        use_columns = data.apply(lambda row: eval(row["Inputs"]), axis=1).tolist()
        print(f"Retrain inputs: {len(use_columns)}")
    else:
        print("No Retrain.csv found")
        sys.exit()
    return use_columns


def load_combinations_csv(path) -> tuple:
    if os.path.exists(path):
        data = pd.read_csv(path)
        comb = data.apply(lambda row: eval(row["Inputs"]), axis=1).tolist()
        comb_columns = [inner_list for outer_list in comb for inner_list in outer_list]

        print(f"Combinations: {len(comb)}")
    else:
        print("No Retrain.csv found")
        sys.exit()
    return comb, comb_columns


def load_input_weight(path, sort: bool = False):
    if os.path.exists(path):
        data = pd.read_csv(path)
        if sort:
            data.sort_values("MAE", axis=0, inplace=True)
        weight = data["RMSE"].to_list()
    else:
        print("No Retrain.csv found")
        sys.exit()
    return weight


def load_model_id(path, sort: bool = False):
    if os.path.exists(path):
        data = pd.read_csv(path)
        if sort:
            data.sort_values("MAE", axis=0, inplace=True)
        model_id = data["Model_id"].to_list()
    else:
        print("No Retrain.csv found")
        sys.exit()
    return model_id


def check_duplicated_prefix_d(lst):
    for word in lst:
        if word[0] == "d" and word[1:] in lst:
            return True
    return False


def filter_combinations(lst, saved_inputs):
    lst = [row for row in lst if not check_duplicated_prefix_d(row)]
    lst = [row for row in lst if not row in saved_inputs]
    print(f"Filtered combinations: {len(lst)}")
    return lst


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_retrain_paths() -> str:
    create_directory(myconfig.RETRAIN_SAVE_PATH)
    main_save_loc = rf"{myconfig.RETRAIN_SAVE_PATH}\All_data_metrics_retrain.csv"
    saveMeanMetrics_Init(main_save_loc)
    return main_save_loc


def process_train_paths() -> str:
    create_directory(myconfig.SAVE_FOLDER)
    main_save_loc = rf"{myconfig.SAVE_FOLDER}\All_data_metrics_main.csv"
    saveModelStructureInit(myconfig.SAVE_FOLDER)
    saveMeanMetrics_Init(main_save_loc)
    return main_save_loc


def process_ensemble_paths() -> str:
    create_directory(myconfig.ENSEMBLE_SAVE_PATH)
    main_save_loc = rf"{myconfig.ENSEMBLE_SAVE_PATH}\All_data_metrics_ensemble.csv"
    saveMeanMetrics_Init(main_save_loc)
    return main_save_loc


def find_best_fold_by_mae(results) -> tuple:
    metrics_f = [val for val in zip(*results)]
    return metrics_f[1].index(min(metrics_f[1]))


def save_best_fold_names(fold, train_name_list, test_name_list, experiments) -> None:
    lists = list()
    experiments = experiments.tolist()
    lists.append(
        [
            experiments.index(item)
            for item in train_name_list[fold]
            if item in experiments
        ]
    )
    lists.append(
        [
            experiments.index(item)
            for item in test_name_list[fold]
            if item in experiments
        ]
    )
    df = pd.DataFrame({"lists": lists})
    df.to_csv(f"{myconfig.SHEETS_PATH}.csv", index=False)


def generate_input_combinations(
    items, label, first_element: str = None, min_length: int = 1, max_length: int = None
):
    """
    Returns column combinations

    Args:
    items (list): A list of items.
    first_element (str): The first element to use in the combinations.
    label (str): The last element to use in the combinations.
    min_length (int): The minimum length of the combinations without label.

    Returns:
    combinations (list): A list of combinations, where each combination is a list of item names.
    """
    if max_length is None:
        max_length = len(items)
    # Remove the last element from the list
    items.remove(label)
    # Remove the first element from the list if it's not None
    if first_element is not None:
        items.remove(first_element)

    # Generate all possible combinations of items
    combinations = []
    for i in range(min_length, max_length):
        for comb in itertools.combinations(items, i):
            if first_element is not None:
                comb = [first_element] + list(comb)

            comb = list(comb) + [label]
            if comb not in combinations and len(comb) > min_length:
                combinations.append(comb)

    print(f"Combinations: {len(combinations)}")
    return combinations


def weighted_predictions(weight_list, prediction_list) -> tuple:
    weight_arr = np.array(weight_list)
    pred_arr = np.array(prediction_list)

    weights = (weight_arr.sum() - weight_arr) / (
        weight_arr.sum() * (len(weight_arr) - 1)
    )
    weighted_predictions = pred_arr * weights[:, None]
    return weighted_predictions.sum(axis=0).tolist(), weights.tolist()


def mean_predictions(prediction_list) -> tuple:
    pred_arr = np.array(prediction_list)

    return pred_arr.mean(axis=0).tolist()


def extract_predictions(df, combination):
    weight_list = []
    prediction_list = []

    for input_value in combination:
        input_series = pd.Series(input_value)
        if any(input_series.equals(pd.Series(row)) for row in df["Inputs"]):
            selected_rows = df[
                df["Inputs"].apply(lambda x: pd.Series(x).equals(input_series))
            ]
            weight_list.extend(selected_rows["Weight"].tolist())
            prediction_list.extend(selected_rows["Prediction"].tolist())

    return weight_list, prediction_list


def residual_sum_of_squares(y_true, y_pred):
    return np.sum(np.square((np.abs(y_true - y_pred))))


def modified_mean_square_error(y_true, y_pred):
    return np.mean(np.square(np.abs(y_true - y_pred) + 1))


def mean_square_error(y_true, y_pred):
    return np.mean(np.square(np.abs(y_true - y_pred)))


def root_mean_square_error(y_true, y_pred):
    return np.sqrt((np.mean(np.square(np.abs(y_true - y_pred)))))


def calculateMetrics(y_true, y_pred) -> tuple:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    """
    returns:
    list of 'MMSE', 'MAE', 'MSE', 'RMSE', 'RSS', 'R2'
    """
    mae = round(mean_absolute_error(y_true=y_true_arr, y_pred=y_pred_arr), 8)
    rss = round(residual_sum_of_squares(y_true=y_true_arr, y_pred=y_pred_arr), 8)
    mmse = round(modified_mean_square_error(y_true=y_true_arr, y_pred=y_pred_arr), 8)
    mse = round(mean_square_error(y_true=y_true_arr, y_pred=y_pred_arr), 8)
    r2 = round(r2_score(y_true=y_true_arr, y_pred=y_pred_arr), 8)
    rmse = round(root_mean_square_error(y_true=y_true_arr, y_pred=y_pred_arr), 8)

    metrics_list = [mmse, mae, mse, rmse, rss, r2]

    return metrics_list


def calculateMeanMetrics(y_true=None, y_pred=None, metrics=None):
    if metrics != None:
        mean_values = [sum(sublist) / len(sublist) for sublist in metrics]
        rounded_values = [round(x, 8) for x in mean_values]
        return rounded_values
    elif y_true != None and y_pred != None:
        mean_values = [
            sum(sublist) / len(sublist)
            for sublist in calculateMetrics(y_true=y_true, y_pred=y_pred)
        ]
        rounded_values = [round(x, 8) for x in mean_values]
        return rounded_values
    else:
        print("Wrong values")

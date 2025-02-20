import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Or "2" to suppress INFO messages

import itertools
import math
import pandas as pd
import tensorflow as tf
from tqdm import tqdm as pbar

# Use TensorFlow's Keras for consistency:
from tensorflow import keras

# Import your modules (ensure each folder has an __init__.py file):
import data.data_preparation as dataprep
import visualization.graphs as graphs
import models.model_architecture as model_architecture
import utils.utils as utils
from data.data_cleaning import Load_Clean_BTPH, Load_Clean_FERM, Load_Clean_GSK
from utils.myconfig import myconfig

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.options.mode.chained_assignment = None

### SEARCH
# 1. load:
#       - Retrained inputs
#       - weights
#       - model_id
#       - data
# 2. Make predictions
# 3. Save best prediction
# 4. Iterate searching


def load_data():
    if myconfig.LOAD == "FERM":
        return Load_Clean_FERM(load_clean=True)
    elif myconfig.LOAD == "BTPH":
        return Load_Clean_BTPH(load_clean=True)
    elif myconfig.LOAD == "GSK":
        return Load_Clean_GSK(load_clean=True)


def make_prediction(model, valid_x_list):
    # Update the custom objects dictionary with your custom functions
    custom_objs = tf.keras.utils.get_custom_objects()
    custom_objs["MMSE_loss"] = model_architecture.MMSE_loss
    custom_objs["RSS_metric"] = model_architecture.RSS_metric
    custom_objs["coeff_determination"] = model_architecture.coeff_determination

    # Now that custom objects are registered, predict as usual
    return model.predict(valid_x_list).flatten()


if __name__ == "__main__":
    main_save_loc = utils.process_ensemble_paths()

    use_columns = utils.load_inputs_csv(myconfig.RETRAIN_SAVE_METRICS_CSV, sort=True)
    weights = utils.load_input_weight(myconfig.RETRAIN_SAVE_METRICS_CSV, sort=True)
    model_id = utils.load_model_id(myconfig.RETRAIN_SAVE_METRICS_CSV, sort=True)

    dataPool = load_data()

    predictions_list = []
    validation_list = []

    train_name_list, test_name_list = dataprep.read_experiment_names(
        experiments=dataPool["Sheet"].unique()
    )

    data_df = pd.DataFrame(columns=["Inputs", "Weight", "Prediction", "Validation"])
    # _______________Predict______________________________
    for id, inputs in enumerate(pbar(use_columns)):
        print(inputs)

        (
            train_x_list,
            train_y_list,
            valid_x_list,
            valid_y_list,
            valid_time,
        ) = dataprep.split_train_test(dataPool, inputs, train_name_list, test_name_list)

        model = tf.keras.models.load_model(
            rf"{myconfig.MODEL_SAVE_PATH}\{model_id[id]}"
        )
        predictions = make_prediction(model, valid_x_list)

        row = [inputs, weights[id], predictions, valid_y_list[0]]
        data_df.loc[len(data_df)] = row

    combination = list()
    best_combination_all_time = [data_df["Inputs"].loc[0]]
    best_combination_temp = best_combination_all_time
    best_metrics = utils.calculateMetrics(
        data_df["Validation"].loc[0], data_df["Prediction"].loc[0]
    )
    best_weighted_prediction = []
    best_weights = []
    best_mae = best_metrics[1]
    print(best_mae)
    id = 0
    add = 0
    utils.saveMeanMetrics(
        main_save_loc, [id, best_combination_all_time, *best_metrics, 0, 1]
    )
    graphs.createSaveGraphs(
        save_name=f"{myconfig.ENSEMBLE_SAVE_PATH}\{0}.svg",
        mode="",
        read_columns=data_df["Inputs"].loc[0],
        plot_valid_y=data_df["Validation"].loc[0],
        plot_test_predictions=data_df["Prediction"].loc[0],
        valid_time=valid_time[0],
    )

    for i in pbar(range(len(use_columns))):
        print(f"Iteration - {i}; State - {add}")
        lst = [x for x in use_columns if x not in best_combination_all_time]
        combinations = itertools.combinations(lst, 1 + add)
        print(f"Combinations: {math.comb(len(lst), 1+add)}")
        for comb in combinations:
            # print(add)
            combination = best_combination_all_time + list(comb)
            # print(combination)
            # print(combination)
            weight_list, prediction_list = utils.extract_predictions(
                data_df, combination
            )
            weighted_prediction, weights = utils.weighted_predictions(
                weight_list, prediction_list
            )

            metrics = utils.calculateMetrics(valid_y_list[0], weighted_prediction)
            if metrics[1] < best_mae:
                best_metrics = metrics
                best_mae = metrics[1]
                best_weighted_prediction = weighted_prediction
                best_weights = weights
                best_combination_temp = combination.copy()

        if best_combination_all_time != best_combination_temp:
            id += 1
            best_combination_all_time = best_combination_temp
            print(f"MAE - {best_mae}; {best_combination_all_time}")
            utils.saveMeanMetrics(
                main_save_loc,
                [id, best_combination_all_time, *best_metrics, 0, best_weights],
            )
            graphs.createSaveGraphs(
                save_name=f"{myconfig.ENSEMBLE_SAVE_PATH}\{id}.svg",
                mode="",
                read_columns=inputs,
                plot_valid_y=valid_y_list[0],
                plot_test_predictions=best_weighted_prediction,
                valid_time=valid_time[0],
            )
            add = 0
        else:
            if add < 5:
                add += 1
            else:
                add = 0

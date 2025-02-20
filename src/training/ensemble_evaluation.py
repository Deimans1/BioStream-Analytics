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


def load_data():
    if myconfig.LOAD == "FERM":
        return Load_Clean_FERM(load_clean=True)
    elif myconfig.LOAD == "BTPH":
        return Load_Clean_BTPH(load_clean=True)
    elif myconfig.LOAD == "GSK":
        return Load_Clean_GSK(load_clean=True)


# def make_prediction(model, valid_x_list):
#     with custom_object_scope(
#         {
#             "MMSE_loss": model_architecture.MMSE_loss,
#             "RSS_metric": model_architecture.RSS_metric,
#             "coeff_determination": model_architecture.coeff_determination,
#         }
#     ):
#         return (model.predict(valid_x_list)).flatten()

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
    combinations, comb_columns = utils.load_combinations_csv(
        myconfig.ENSEMBLE_LOAD_PATH
    )

    dataPool = load_data()

    train_name_list, test_name_list = dataprep.read_experiment_names(
        dataPool["Sheet"].unique()
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

        predictions = []
        for idx, valid_x in enumerate(valid_x_list):
            model = tf.keras.models.load_model(
                rf"{myconfig.MODEL_SAVE_PATH}\{model_id[id]}_{idx}"
            )
            predictions.append(make_prediction(model, valid_x))

        row = [inputs, weights[id], predictions, valid_y_list[0]]
        data_df.loc[len(data_df)] = row

    data_df = data_df[data_df["Inputs"].isin(comb_columns)]
    data_df.reset_index(drop=True, inplace=True)

    id = 0

    for comb in combinations:
        weight_list, prediction_list = utils.extract_predictions(data_df, comb)
        if len(comb) < 2:
            weighted_prediction = prediction_list[0]
        else:
            weighted_prediction, weights = utils.weighted_predictions(
                weight_list, prediction_list
            )

        metrics = utils.calculateMetrics(
            data_df["Validation"].loc[0], weighted_prediction
        )
        best_metrics = metrics
        best_mae = metrics[1]
        best_weighted_prediction = weighted_prediction
        best_weights = weights

        print(f"MAE - {best_mae}; {comb}")
        utils.saveMeanMetrics(
            main_save_loc,
            [id, comb, *best_metrics, 0, best_weights],
        )
        graphs.createSaveGraphs(
            save_name=f"{myconfig.ENSEMBLE_SAVE_PATH}\{id}.svg",
            mode="",
            read_columns=inputs,
            plot_valid_y=data_df["Validation"].loc[0],
            plot_test_predictions=best_weighted_prediction,
            valid_time=valid_time[0],
        )
        id += 1

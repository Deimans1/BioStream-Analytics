import csv
import itertools
import multiprocessing
import os
import sys
import time

import pandas as pd
from tqdm import tqdm as pbar

import data.data_preparation as dataprep
import visualization.graphs as graphs
import models.model_architecture as model_architecture
import utils.utils as utils
from data.data_cleaning import Load_Clean_BTPH, Load_Clean_FERM, Load_Clean_GSK
from utils.myconfig import myconfig

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.options.mode.chained_assignment = None


if __name__ == "__main__":
    best_fold_index = None
    if myconfig.LOAD == "FERM":
        dataPool = Load_Clean_FERM(load_clean=True)
    elif myconfig.LOAD == "BTPH":
        dataPool = Load_Clean_BTPH(load_clean=True)
    elif myconfig.LOAD == "GSK":
        dataPool = Load_Clean_GSK(load_clean=True)

    if myconfig.RETRAIN:
        main_save_loc = utils.process_retrain_paths()
        use_columns = utils.LoadTrainedInputs(myconfig.RETRAIN_METRICS_CSV)
        print(len(use_columns))

    for id, inputs in enumerate(pbar(use_columns)):
        print(inputs)

        train_name_list, test_name_list = dataprep.kfold_experiments(
            kfold=myconfig.kfold, experiments=dataPool["Sheet"].unique()
        )

        (
            train_x_list,
            train_y_list,
            valid_x_list,
            valid_y_list,
            valid_time,
        ) = dataprep.split_train_test(dataPool, inputs, train_name_list, test_name_list)
        model_list = dataprep.build_models(train_x_list)
        print("model_list_len: ", len(model_list))

        cross_time = time.time()
        with multiprocessing.Pool(processes=myconfig.CORES) as pool:
            retrieved = pool.starmap(
                model_architecture.ModelFit,
                zip(
                    model_list,
                    train_x_list,
                    train_y_list,
                    valid_x_list,
                    valid_y_list,
                    itertools.repeat(myconfig.callback),
                ),
            )

        print(
            f"Cross Validation time: {round((time.time()-cross_time),4)} s | {round((time.time()-cross_time)/60,2)} min"
        )

        results, predictions, model_list = zip(*retrieved)

        metrics = list()
        for val in zip(*results):
            metrics.extend(val)
        with open(
            rf"{myconfig.RETRAIN_SAVE_PATH}\metrics.csv", mode="a+", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(metrics)

        metrics_average = [sum(val) / len(val) for val in zip(*results)]
        metrics_long = [id, inputs, *metrics_average, id]

        utils.saveMeanMetrics(main_save_loc, metrics_long)
        print(metrics_average)

        if myconfig.RETRAIN:
            if id == 0:
                best_fold_index = utils.find_best_fold_by_mae(results)
                print(best_fold_index)
                utils.save_best_fold_names(
                    best_fold_index,
                    train_name_list,
                    test_name_list,
                    experiments=dataPool["Sheet"].unique(),
                )

            if id < myconfig.NUM_OF_IMG_TO_SAVE:
                for idx, prediction in enumerate(predictions):
                    if idx == best_fold_index or id > 0:
                        graphs.createSaveGraphs(
                            save_name=f"{myconfig.RETRAIN_SAVE_PATH}\{id}_{idx}.svg",
                            mode="",
                            read_columns=inputs,
                            plot_valid_y=valid_y_list[idx],
                            plot_test_predictions=prediction,
                            valid_time=valid_time[idx],
                        )

            if myconfig.SAVE_MODELS:
                for idx, model in enumerate(model_list):
                    model.save(rf"{myconfig.MODEL_SAVE_PATH}\{id}_{idx}")

            if myconfig.STOP_AFTER_IMG_SAVE and id == myconfig.NUM_OF_IMG_TO_SAVE:
                sys.exit()

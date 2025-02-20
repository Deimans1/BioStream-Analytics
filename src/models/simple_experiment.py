import itertools
import multiprocessing
import os
import sys
import time

import pandas as pd
from tqdm import tqdm as pbar

import data.data_preparation as dataprep
import visualization.graphs as graphs
import model_architecture
import utils
from data.data_cleaning import Load_Clean_BTPH, Load_Clean_FERM, Load_Clean_GSK
from utils.myconfig import myconfig

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.options.mode.chained_assignment = None


if __name__ == "__main__":
    best_fold_index = None

    dataPool = Load_Clean_FERM(load_clean=True)
    main_save_loc = (
        rf"D:\Gitlab\machine-learning\Acetate\data\final\Test\All_data_metrics_main.csv"
    )
    use_columns = [
        [
            "Age",
            "BiomassEst",
            "Carbon_Feed",
            "Induction",
            "miuSimp",
            "Protein",
        ]
    ]

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

        metrics_average = [sum(val) / len(val) for val in zip(*results)]
        metrics_long = [id, inputs, *metrics_average, id]

        utils.saveMeanMetrics(main_save_loc, metrics_long)
        print(metrics_average)

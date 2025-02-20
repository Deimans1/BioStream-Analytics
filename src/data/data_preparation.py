import time
from typing import List

import numpy as np
import pandas as pd

import models.model_architecture as model_architecture
import utils
from utils.myconfig import myconfig


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]

        X.append(seq_x)
        y.append(seq_y)
    return X, y


def prep_data_for_lstm(
    all_data_df: pd.DataFrame,
    excel_columns: List[str] = ["Time", "Biomass", "Protein"],
    n_steps: int = 23,
) -> tuple:
    data_X, data_y = list(), list()

    zeros1 = [0.0 for i in range(len(excel_columns))]
    zeroes = [zeros1 for i in range(n_steps - 1)]

    for sheet_name in all_data_df["Sheet"].unique():
        sheet = all_data_df[(all_data_df["Sheet"] == sheet_name)]

        df = pd.DataFrame(zeroes, columns=excel_columns)
        df = pd.concat([df, sheet], join="inner")
        df = df.tail(n_steps - 1 + len(df.index))

        dataset = df.to_numpy()

        X, y = split_sequences(dataset, n_steps)
        data_X = data_X + X
        data_y = data_y + y

    return np.asarray(data_X), np.asarray(data_y)


def preprocess_data(data_train, data_valid, inputs) -> tuple:
    train_scaled = myconfig.scaler.fit_transform(data_train[inputs])
    valid_scaled = myconfig.scaler.transform(data_valid[inputs])

    train_df = pd.DataFrame(train_scaled, columns=inputs)
    train_df["Sheet"] = data_train["Sheet"]
    valid_df = pd.DataFrame(valid_scaled, columns=inputs)
    valid_df["Sheet"] = data_valid["Sheet"]

    train_x, train_y = prep_data_for_lstm(train_df, inputs, myconfig.N_STEPS)
    valid_x, valid_y = prep_data_for_lstm(valid_df, inputs, myconfig.N_STEPS)

    return train_x, train_y, valid_x, valid_y


def kfold_experiments(kfold, experiments: np.ndarray) -> tuple:
    train_list, test_list = list(), list()

    for train_set, test_set in kfold.split(experiments):
        train_list.append(experiments[train_set].tolist())
        test_list.append(experiments[test_set].tolist())

    return train_list, test_list


def read_experiment_names(experiments: np.ndarray) -> tuple:
    sheets_df = pd.read_csv(rf"{myconfig.SHEETS_PATH}.csv")

    train_list = [experiments[eval(sheets_df["lists"].loc[0])]]
    test_list = [experiments[eval(sheets_df["lists"].loc[1])]]

    return train_list, test_list


def extract_experiments_from_df(df: pd.DataFrame, experiments: list) -> tuple:
    return df[df["Sheet"].isin(experiments)].reset_index(drop=True)


def split_train_test(
    dataPool: pd.DataFrame, features: list, train_list: list, test_list: list
) -> tuple:
    train_x_list, train_y_list, valid_x_list, valid_y_list = (
        list(),
        list(),
        list(),
        list(),
    )
    valid_time = list()

    for train, test in zip(train_list, test_list):
        data_train = extract_experiments_from_df(dataPool, train)
        data_valid = extract_experiments_from_df(dataPool, test)

        train_x, train_y, valid_x, valid_y = preprocess_data(
            data_train, data_valid, features
        )

        train_x_list.append(train_x)
        train_y_list.append(train_y)
        valid_x_list.append(valid_x)
        valid_y_list.append(valid_y)

        valid_time.append(data_valid[["Time", "Sheet"]])

    return train_x_list, train_y_list, valid_x_list, valid_y_list, valid_time


def build_models(train_x_list: list) -> tuple:
    model_list = list()
    for train_x in train_x_list:
        model = model_architecture.build_and_compile_model_LSTM(
            n_steps=train_x.shape[1], n_features=train_x.shape[2]
        )
        model_list.append(model)

    return model_list


def preprocess_and_dataload_kfold(
    kfold, dataPool, inputs, scaler, best_fold=None
) -> tuple:
    data_prep_time = time.time()
    experiments = dataPool["Sheet"].unique()
    train_x_list, train_y_list, valid_x_list, valid_y_list, valid_time, model_list = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )
    sheets = list()

    for idx, (train, valid) in enumerate(kfold.split(experiments)):
        if best_fold is not None:
            if idx == best_fold:
                sheets.append(train)
                sheets.append(valid)

        data_train = dataPool[
            dataPool["Sheet"].isin(experiments[train].tolist())
        ].reset_index(drop=True)
        data_valid = dataPool[
            dataPool["Sheet"].isin(experiments[valid].tolist())
        ].reset_index(drop=True)

        train_x, train_y, valid_x, valid_y = preprocess_data(
            data_train, data_valid, inputs, scaler
        )
        # print(train_x.shape, train_y.shape)
        # print(valid_x.shape, valid_y.shape)

        train_x_list.append(train_x)
        train_y_list.append(train_y)
        valid_x_list.append(valid_x)
        valid_y_list.append(valid_y)

        valid_time.append(data_valid[["Time", "Sheet"]])

        model, custom_optimizer = model_architecture.build_and_compile_model_LSTM(
            n_steps=train_x.shape[1], n_features=train_x.shape[2]
        )
        utils.saveModelStructure_Model(myconfig.SAVE_FOLDER, model, custom_optimizer)
        model_list.append(model)

    print(
        f"Data preparation time: {round((time.time()-data_prep_time),4)} s | {round((time.time()-data_prep_time)/60,2)} min"
    )

    return (
        model_list,
        train_x_list,
        train_y_list,
        valid_x_list,
        valid_y_list,
        valid_time,
        sheets,
    )


def preprocess_and_dataload(dataPool, inputs, scaler, sheets_path) -> tuple:
    data_prep_time = time.time()
    experiments = dataPool["Sheet"].unique()
    train_x_list, train_y_list, valid_x_list, valid_y_list, valid_time, model_list = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )

    sheets_df = pd.read_csv(rf"{sheets_path}.csv")
    print(sheets_df["lists"].loc[0])
    print(type(eval(sheets_df["lists"].loc[1])))

    data_train = dataPool[
        dataPool["Sheet"].isin(experiments[eval(sheets_df["lists"].loc[0])].tolist())
    ].reset_index(drop=True)
    data_valid = dataPool[
        dataPool["Sheet"].isin(experiments[eval(sheets_df["lists"].loc[1])].tolist())
    ].reset_index(drop=True)

    train_x, train_y, valid_x, valid_y = preprocess_data(
        data_train, data_valid, inputs, scaler
    )
    # print(train_x.shape, train_y.shape)
    # print(valid_x.shape, valid_y.shape)

    train_x_list.append(train_x)
    train_y_list.append(train_y)
    valid_x_list.append(valid_x)
    valid_y_list.append(valid_y)

    valid_time.append(data_valid[["Time", "Sheet"]])

    model, _ = model_architecture.build_and_compile_model_LSTM(
        n_steps=train_x.shape[1], n_features=train_x.shape[2]
    )
    model_list.append(model)

    print(
        f"Data preparation time: {round((time.time()-data_prep_time),4)} s | {round((time.time()-data_prep_time)/60,2)} min"
    )

    return (
        model_list,
        train_x_list,
        train_y_list,
        valid_x_list,
        valid_y_list,
        valid_time,
    )

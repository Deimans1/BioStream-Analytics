import pandas as pd

from myconfig import myconfig

data = pd.read_csv(myconfig.MAIN_SAVE_METRICS_CSV)
data.drop(columns="Test_ID", inplace=True)
print(data.shape)
data.sort_values("MAE", axis=0, inplace=True)
savedInputs = data.apply(lambda row: eval(row["Inputs"]), axis=1).tolist()

for n, row in enumerate(savedInputs[:-1]):
    for idx, row2 in enumerate(savedInputs[n + 1 :]):
        if set(row).issubset(set(row2)) or set(row2) <= set(row):
            savedInputs.remove(row2)

print("Remaining lists:")
savedInputsSTR = [repr(sublist) for sublist in savedInputs]

# Filter the DataFrame based on matching sublists
filtered_df = data[data["Inputs"].isin(savedInputsSTR)]
filtered_df.reset_index(drop=True, inplace=True)
print(filtered_df.shape)

mean_mae = filtered_df["MAE"].mean()
quart = filtered_df["MAE"].quantile(myconfig.FILTER_QUANTILE)  # 0.25
mask = filtered_df["MAE"] >= quart
filtered_df = filtered_df[~mask]
print(filtered_df.shape)
filtered_df.to_csv(f"{myconfig.SAVE_FOLDER}\Filtered_data.csv", mode="w", index=False)

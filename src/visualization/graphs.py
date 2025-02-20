import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.use("SVG")


# --------------Graphics----------------------------
def plot_validation_subplot(time, valid, predict, read_columns, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(time, predict, label="Predicted data")
    ax.scatter(time, valid, label="Original data")
    ax.grid(color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Time, h")
    ax.set_ylabel(f"{read_columns[-1]}, g/L")
    ax.legend(loc=2, fontsize=12)


def plot_validation(
    plot_valid_y,
    plot_test_predictions,
    valid_time,
    read_columns,
    title: str = "Validation plot",
):
    time_arr = np.array(valid_time["Time"])
    sheet_arr = np.array(valid_time["Sheet"])
    indices = np.where(sheet_arr[1:] != sheet_arr[:-1])[0]
    sub_time = np.split(time_arr, indices + 1)
    sub_val = np.split(plot_valid_y, indices + 1)
    sub_pred = np.split(plot_test_predictions, indices + 1)

    # num_plots = len(sub_time) - 1
    # if num_plots > 6:
    #     num_plots = 6

    num_plots = 6
    num_rows, num_cols = divmod(num_plots, 3)
    if (num_cols + 1) > 0:
        num_rows += 1

    fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(14, 10))
    for i, ax in enumerate(axs.flat):
        if i + 1 <= len(sub_time) and i < ((num_rows * 3) - 1):
            plot_validation_subplot(
                sub_time[i], sub_val[i], sub_pred[i], ax=ax, read_columns=read_columns
            )
            ax.set_title(f"Validation {i+1}", fontsize=12)
        else:
            ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    return fig, axs


def createSaveGraphs(
    save_name, plot_valid_y, plot_test_predictions, valid_time, mode, read_columns
):
    title = mode + ": " + ", ".join(read_columns)

    fig, axs = plot_validation(
        plot_valid_y,
        plot_test_predictions,
        valid_time,
        title=title,
        read_columns=read_columns,
    )

    if isinstance(axs, np.ndarray) and axs.ndim == 2:
        gs = axs[0, 0].get_gridspec()
    else:
        gs = axs[0].get_gridspec()

    num_rows, num_cols = gs.get_geometry()
    num_subplots = len(axs.flat)

    ax_last = fig.add_subplot(num_rows, num_cols, num_subplots)
    ax_last.scatter(plot_valid_y, plot_test_predictions)
    ax_last.set_xlabel("True Values [" + read_columns[-1] + "]")
    ax_last.set_ylabel("Predictions [" + read_columns[-1] + "]")
    lims = [
        min(plot_valid_y) - min(plot_valid_y),
        max(plot_valid_y) + (max(plot_valid_y) * 0.5),
    ]
    ax_last.set_xlim(lims)
    ax_last.set_ylim(lims)
    ax_last.plot(lims, lims)
    plt.tight_layout()

    print("Saving images....")
    fig.savefig(save_name, format="svg", dpi=600)

    print("Images saved")

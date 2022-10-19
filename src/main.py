from numbers import Number
from typing import Collection
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto

from column import DURATION_COL
from data import get_time_log_df
from path import get_csv_path

SEC_PER_MIN = 60
EPSILON = 1e-10


def show_durations_follow_pareto(
    duration_mins: Collection[Number],
    max_plot_mins: Number = 400,
) -> None:
    dist = pareto
    args = dist.fit(duration_mins)
    # Plot the fit and empirical CDF
    x = np.linspace(0, max_plot_mins, 1000)
    y = dist.cdf(x, *args)
    _, ax = plt.subplots()
    ax.plot(x, y, label="Pareto fit")
    ax.hist(
        duration_mins,
        density=True,
        cumulative=True,
        label="Empirical CDF",
        bins=1000,
        histtype="step",
    )
    ax.set_xlim(-15, max_plot_mins)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Duration (mins)")
    ax.set_ylabel("Cumulative probability")
    ax.legend()
    plt.plot()


if __name__ == "__main__":
    csv_path = get_csv_path()
    df = get_time_log_df(csv_path)

    duration_mins = df[DURATION_COL].dt.seconds / SEC_PER_MIN
    # near-0 minutes gets rounded down so exclude
    duration_mins[duration_mins == 0] = EPSILON
    show_durations_follow_pareto(duration_mins)

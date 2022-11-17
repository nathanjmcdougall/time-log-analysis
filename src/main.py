from numbers import Number
from typing import Collection
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
import datetime
import random
from matplotlib.ticker import PercentFormatter

from column import DURATION_COL, FROM_COL, TO_COL, ACTIVITY_COL
from data import get_time_log_df
from path import get_csv_path

SEC_PER_MIN = 60
MIN_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7

MIN_PER_DAY = HOURS_PER_DAY * MIN_PER_HOUR
MIN_PER_WEEK = DAYS_PER_WEEK * MIN_PER_DAY

EPSILON = 1e-10

# Set font size to 22
plt.rcParams.update({"font.size": 22})


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

    nonzero_df = df.loc[df[DURATION_COL].dt.seconds > 0].copy()
    nonzero_df["diff"] = nonzero_df[FROM_COL] - nonzero_df[TO_COL].shift(1)
    nonzero_df["diff_secs"] = nonzero_df["diff"].dt.seconds
    nonzero_df.reset_index(drop=True, inplace=True)
    bad_idx = nonzero_df["diff_secs"] > SEC_PER_MIN * 1
    bad_df = nonzero_df.loc[bad_idx]
    bad_indexes = bad_df.index
    for bad_index in bad_indexes:
        bad_time = nonzero_df.loc[bad_index, FROM_COL]
        bad_gap = int(nonzero_df.loc[bad_index, "diff_secs"] / SEC_PER_MIN)
        warnings.warn(f"Gap of {bad_gap}min: {bad_time}")

    gap_idx = nonzero_df["diff_secs"] > 0
    gap_df = nonzero_df.loc[gap_idx]
    for gap_index in gap_df.index:
        gap = nonzero_df.loc[gap_index, "diff"]
        midway_time = nonzero_df.loc[gap_index, FROM_COL] + gap / 2
        nonzero_df.loc[gap_index, FROM_COL] = midway_time
        nonzero_df.loc[gap_index - 1, TO_COL] = midway_time

    nonzero_df["diff"] = nonzero_df[FROM_COL] - nonzero_df[TO_COL].shift(1)
    nonzero_df["diff_secs"] = nonzero_df["diff"].dt.seconds

    assert not (nonzero_df["diff_secs"] > 0).any()

    nonzero_df["From Day"] = nonzero_df[FROM_COL].dt.date
    nonzero_df["To Day"] = nonzero_df[TO_COL].dt.date

    # Split rows spanning acrosss multiple days into single rows
    spanning_idx = nonzero_df["From Day"] != nonzero_df["To Day"]
    spanning_rows = nonzero_df[spanning_idx]
    nonspanning_rows = nonzero_df[~spanning_idx]
    from_part = spanning_rows.copy()
    to_part = spanning_rows.copy()
    from_part[FROM_COL] = from_part[TO_COL].apply(
        lambda x: x.replace(hour=0, minute=0, second=0)
    )
    to_part[TO_COL] = to_part[FROM_COL].apply(
        lambda x: x.replace(hour=23, minute=59, second=59)
    )

    nonzero_df = pd.concat([nonspanning_rows, from_part, to_part], ignore_index=True)
    nonzero_df[DURATION_COL] = nonzero_df[TO_COL] - nonzero_df[FROM_COL]
    nonzero_df.sort_values(by=[FROM_COL], inplace=True)
    nonzero_df.reset_index(drop=True, inplace=True)

    nonzero_df.drop(columns=["diff", "diff_secs", "From Day", "To Day"], inplace=True)

    plot_df = nonzero_df.copy()
    # plotting range
    start_date = datetime.date(2022, 3, 12)
    end_date = datetime.date.today()

    plot_df = plot_df.loc[
        (plot_df[FROM_COL].dt.date >= start_date)
        & (plot_df[TO_COL].dt.date <= end_date)
    ]

    # find the top activities
    n_colors = 20
    activity_counts = plot_df.groupby(ACTIVITY_COL)[DURATION_COL].sum()
    activity_counts.sort_values(ascending=False, inplace=True)
    top_activities = activity_counts.index[: n_colors - 1]
    plot_df[ACTIVITY_COL] = plot_df[ACTIVITY_COL].apply(
        lambda x: x if x in top_activities else "Other"
    )

    # Plot the span of each activity by day
    # every activity needs a different colour so we need a custom matplotlib
    # colour scale
    cmap = plt.get_cmap("tab20")
    colors = cmap(range(n_colors))

    fig, ax = plt.subplots(figsize=(40, 10))
    for color, activity in zip(colors, top_activities.tolist() + ["Other"]):
        this_df = plot_df.loc[plot_df[ACTIVITY_COL] == activity]
        bottom = this_df[FROM_COL].dt.hour * MIN_PER_HOUR + this_df[FROM_COL].dt.minute
        plt.bar(
            this_df[FROM_COL].dt.date,
            this_df[DURATION_COL].dt.seconds / SEC_PER_MIN,
            width=1,
            bottom=bottom,
            label=activity,
            color=color,
        )

    # Y-axis labels
    # start with 12am at the top and then format times as 12:00am, 1:00am, 2:00am, etc.
    # working downwards
    # invert the actual values
    ax.set_ylim(0, MIN_PER_DAY)
    yticks = np.arange(0, MIN_PER_DAY, 60)
    yticklabels = [f"{i % 12 or 12}:00{'am' if i < 12 else 'pm'}" for i in range(24)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Time of day")
    ax.invert_yaxis()

    # X-axis
    half_day = datetime.timedelta(hours=12)
    ax.set_xlim(start_date - half_day, end_date + half_day)
    ax.set_xlabel("Date")

    # legend to the right of the graph, centred vertically
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Activity")

    plt.savefig("plot_daily.png", bbox_inches="tight", dpi=350)
    plt.show()

    ## Now do the same but don't plot a time but a stacked bar chart showing the average
    ## time spent on every day up to that point
    fig, ax = plt.subplots(figsize=(40, 10))
    templ_plot_df = plot_df.copy()
    MIN_COL = "minutes"
    templ_plot_df[MIN_COL] = templ_plot_df[DURATION_COL].dt.seconds / SEC_PER_MIN
    template_contrib = templ_plot_df.groupby(plot_df[FROM_COL].dt.date)[MIN_COL].sum()
    total_minutes = template_contrib.cumsum()
    bottom = 0
    for color, activity in zip(colors, top_activities.tolist() + ["Other"]):
        this_df = templ_plot_df.loc[plot_df[ACTIVITY_COL] == activity].copy()
        contrib = this_df.groupby(this_df[FROM_COL].dt.date)[MIN_COL].sum()
        contrib = (
            contrib.reindex(template_contrib.index, fill_value=0).cumsum()
            / total_minutes
        )
        plt.bar(
            contrib.index,
            contrib,
            width=1,
            bottom=bottom,
            label=activity,
            color=color,
        )
        bottom += contrib

    # Y-axis labels
    # Percentage formatting
    ax.set_ylim(0, 1)
    ytickformatter = PercentFormatter(xmax=1)
    ax.yaxis.set_major_formatter(ytickformatter)
    ax.set_ylabel("Percentage of total time")

    # X-axis
    half_day = datetime.timedelta(hours=12)
    ax.set_xlim(start_date - half_day, end_date + half_day)
    ax.set_xlabel("Date")

    # legend to the right of the graph, centred vertically
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Activity")

    plt.savefig("plot_cumulative.png", bbox_inches="tight", dpi=350)
    plt.show()

    ## Now do the same but instead of a cumulative total from the very beginning, just
    # The last 30 day
    fig, ax = plt.subplots(figsize=(40, 10))
    templ_plot_df = plot_df.copy()
    MIN_COL = "minutes"
    templ_plot_df[MIN_COL] = templ_plot_df[DURATION_COL].dt.seconds / SEC_PER_MIN
    template_contrib = templ_plot_df.groupby(plot_df[FROM_COL].dt.date)[MIN_COL].sum()
    template_contrib.index = pd.to_datetime(template_contrib.index)
    total_minutes = template_contrib.rolling("30D").sum()
    bottom = 0
    top_activities_without_work = top_activities.tolist().copy()
    top_activities_without_work.remove("Work")
    work_idx = top_activities.tolist().index("Work")
    colors_without_work = colors.tolist()
    colors_without_work = (
        colors_without_work[:work_idx] + colors_without_work[work_idx + 1 :]
    )
    synced_colors = colors_without_work + [colors[work_idx]]
    synced_activities = top_activities_without_work + ["Work", "Other"]

    for color, activity in zip(synced_colors, synced_activities):

        this_df = templ_plot_df.loc[plot_df[ACTIVITY_COL] == activity].copy()
        contrib = this_df.groupby(this_df[FROM_COL].dt.date)[MIN_COL].sum()
        contrib = (
            contrib.reindex(template_contrib.index, fill_value=0).rolling("30D").sum()
            / total_minutes
        )
        plt.bar(
            contrib.index,
            contrib,
            width=1,
            bottom=bottom,
            label=activity,
            color=color,
        )
        bottom += contrib

    # Y-axis labels
    # Percentage formatting
    ax.set_ylim(0, 1)
    ytickformatter = PercentFormatter(xmax=1)
    ax.yaxis.set_major_formatter(ytickformatter)
    ax.set_ylabel("Percentage of total time")

    # X-axis
    half_day = datetime.timedelta(hours=12)
    ax.set_xlim(start_date - half_day, end_date + half_day)
    ax.set_xlabel("Date")

    # legend to the right of the graph, centred vertically
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Activity")

    plt.savefig("plot_cumulative_monthly.png", bbox_inches="tight", dpi=350)
    plt.show()

    # Now do the same thing but based on week
    # Plot the span of each activity by week
    fig, ax = plt.subplots(figsize=(20, 20))
    for color, activity in zip(colors, top_activities.tolist() + ["Other"]):
        this_df = plot_df.loc[plot_df[ACTIVITY_COL] == activity]
        bottom = (
            this_df[FROM_COL].dt.dayofweek * MIN_PER_DAY
            + this_df[FROM_COL].dt.hour * MIN_PER_HOUR
            + this_df[FROM_COL].dt.minute
        )
        plt.bar(
            this_df[FROM_COL].dt.isocalendar().week,
            this_df[DURATION_COL].dt.seconds / SEC_PER_MIN,
            width=1,
            bottom=bottom,
            label=activity,
            color=color,
        )

    # Y-axis labels
    # start with Monday at the top and then format days as Mon, Tue, Wed, etc.
    # working downwards
    # invert the actual values
    ax.set_ylim(0, MIN_PER_WEEK)
    yticks = np.arange(0, MIN_PER_WEEK, MIN_PER_DAY)
    yticklabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Day of week")
    ax.invert_yaxis()

    # X-axis
    ax.set_xlim(start_date.isocalendar().week - 1, end_date.isocalendar().week + 1)
    ax.set_xlabel("Week of Year")

    # legend to the right of the graph, centred vertically
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Activity")

    plt.savefig("plot_weekly.png", bbox_inches="tight", dpi=350)
    plt.show()

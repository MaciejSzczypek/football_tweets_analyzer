import datetime
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from configs.config_schema import PathsConfig
from data.column_names import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME, CREATED_AT_COLUMN_NAME
from utils.printing import section_printing_decorator


def _adjust_time_to_frame(time, time_frame_minutes_length):
    """ Adjust time to the nearest start or end time frame. """
    return time.replace(
        second=0,
        minute=math.floor(time.minute / time_frame_minutes_length) * time_frame_minutes_length
    )


def _get_max_time_frame(time, time_frame_minutes_length):
    """ Get the maximum time frame based on the max time in the dataset. """
    max_time_minutes = math.ceil(
        (time.minute + (1 if time.second else 0)) / time_frame_minutes_length) * time_frame_minutes_length
    return time.replace(
        second=0,
        minute=max_time_minutes if max_time_minutes != 60 else 0,
        hour=time.hour + (1 if max_time_minutes == 60 else 0)
    )


def _create_time_frames(min_time, max_time, time_frame_minutes_length):
    """ Generate the time frames between min_time and max_time. """
    time_frames = []
    current_time_frame_start = min_time
    while current_time_frame_start + datetime.timedelta(minutes=time_frame_minutes_length) <= max_time:
        current_time_frame_end = current_time_frame_start + datetime.timedelta(
            minutes=time_frame_minutes_length) - datetime.timedelta(seconds=1)
        time_frames.append((current_time_frame_start, current_time_frame_end))
        current_time_frame_start += datetime.timedelta(minutes=time_frame_minutes_length)
    return time_frames


def _get_time_frames(time_series: pd.Series, time_frame_minutes_length: int) -> pd.DataFrame:
    min_time, max_time = min(time_series), max(time_series)

    min_time_series_time = _adjust_time_to_frame(min_time, time_frame_minutes_length)
    max_time_series_time = _get_max_time_frame(max_time, time_frame_minutes_length)

    time_frames = _create_time_frames(min_time_series_time, max_time_series_time, time_frame_minutes_length)

    df_time_frames = pd.DataFrame(time_frames, columns=["start", "end"])
    df_time_frames["start_hour"] = df_time_frames.start.dt.time
    df_time_frames["end_hour"] = df_time_frames.end.dt.time
    df_time_frames[LABEL_COLUMN_NAME] = df_time_frames["start_hour"].astype(str) + " - " + df_time_frames[
        "end_hour"].astype(str)

    return df_time_frames


def _prepare_figure(df: pd.DataFrame, time_frames, y_label: str, file_name: str, is_stacked_bar_plot: bool = False):
    plot = df.plot.bar(stacked=is_stacked_bar_plot) if is_stacked_bar_plot else df.plot()
    plot.set_xticks(time_frames.index)
    plot.set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    plot.set_xlabel("Time frame", fontsize=15, labelpad=15)
    plot.set_ylabel(y_label, fontsize=15, labelpad=15)
    plot.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=4)

    plt.tight_layout()
    plot.figure.savefig(file_name)
    plt.show()


def _calculate_counts_and_percentages(df, label_column):
    """ Calculate tweet counts and percentages for each time frame and label column. """
    total_tweets_per_time_frame = df.groupby('time_frame')["text"].count().reset_index(name="time_frame_count")     # todo for topic modeling tweet instead of text
    df = df.groupby(["time_frame", label_column])["text"].count().reset_index(name="count")
    df = df.merge(total_tweets_per_time_frame, on="time_frame")
    df["percentage"] = df["count"] / df["time_frame_count"] * 100
    return df


def _prepare_series(df, label_column):
    """ Prepare series of counts and percentages grouped by label column. """
    count_series = {
        f"{label}": df[df[label_column] == label].set_index("time_frame")["count"]
        for label in df[label_column].unique()
    }
    percentage_series = {
        f"{label}": df[df[label_column] == label].set_index("time_frame")["percentage"]
        for label in df[label_column].unique()
    }
    return count_series, percentage_series


def _show_statistics(df: pd.DataFrame, time_frame_minutes_length: int, paths_config: PathsConfig, label_column: str,
                     file_prefix: str):
    """ Generalized function to show sentiment or topic statistics. """
    time_series = pd.to_datetime(df[CREATED_AT_COLUMN_NAME])
    time_frames = _get_time_frames(time_series, time_frame_minutes_length)

    time_frame_labels = time_series.apply(
        lambda time_record:
        time_frames[(time_record >= time_frames["start"]) & (time_record <= time_frames["end"])].index[0]
    )
    df["time_frame"] = time_frame_labels

    df = _calculate_counts_and_percentages(df, label_column)

    count_series, percentage_series = _prepare_series(df, label_column)

    # Plot count and percentage figures
    _prepare_figure(pd.DataFrame(count_series), time_frames, "Tweet count",
                    os.path.join(paths_config.results_dir, f"{file_prefix}_absolute_linear"))
    _prepare_figure(pd.DataFrame(percentage_series), time_frames, "Tweet percentage [%]",
                    os.path.join(paths_config.results_dir, f"{file_prefix}_percentage_linear"))
    _prepare_figure(pd.DataFrame(count_series), time_frames, "Tweet count",
                    os.path.join(paths_config.results_dir, f"{file_prefix}_absolute_bar"), is_stacked_bar_plot=True)
    _prepare_figure(pd.DataFrame(percentage_series), time_frames, "Tweet percentage [%]",
                    os.path.join(paths_config.results_dir, f"{file_prefix}_percentage_bar"), is_stacked_bar_plot=True)


def _show_topic_statistics(df_topic: pd.DataFrame, time_frame_minutes_length: int, paths_config: PathsConfig):
    _show_statistics(df_topic, time_frame_minutes_length, paths_config, 'topic', 'topics')


def _show_sentiment_statistics(df_sentiment: pd.DataFrame, paths_config: PathsConfig, time_frame_minutes_length: int):
    _show_statistics(df_sentiment, time_frame_minutes_length, paths_config, LABEL_COLUMN_NAME, 'sentiment')


@section_printing_decorator("TIME FRAME SENTIMENT AND TOPICS ANALYSIS")
def show_time_frames_analysis(df_sentiment: pd.DataFrame, df_topic: pd.DataFrame, paths_config: PathsConfig,
                              time_frame_minutes_length: int = 30):
    if df_sentiment is not None:
        _show_sentiment_statistics(df_sentiment, paths_config, time_frame_minutes_length)

    if df_topic is not None:
        _show_topic_statistics(df_topic, time_frame_minutes_length, paths_config)

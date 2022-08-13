import datetime
import math

import matplotlib.pyplot as plt
import pandas as pd

from data.column_names import TEXT_COLUMN_NAME, LABEL_COLUMN_NAME, CREATED_AT_COLUMN_NAME
from sentiment.labeler import Labeler
from utils.printing import section_printing_decorator


def _get_time_frames(time_series: pd.Series, time_frame_minutes_length: int) -> pd.DataFrame:
    min_time = min(time_series)
    max_time = max(time_series)
    min_time_series_time = min_time.replace(
        second=0,
        minute=(
                math.floor(min_time.minute / time_frame_minutes_length) * time_frame_minutes_length
        )
    )
    max_time_series_minutes = (
            math.ceil((max_time.minute + (1 if max_time.second else 0)) / time_frame_minutes_length) * time_frame_minutes_length
    )
    max_time_series_time = max_time.replace(
        second=0,
        minute=max_time_series_minutes if max_time_series_minutes != 60 else 0,
        hour=max_time.hour if max_time_series_minutes != 60 else max_time.hour + 1
    )
    current_time_frame_start = min_time_series_time
    time_frames = []
    while (
            current_time_frame_start + datetime.timedelta(minutes=time_frame_minutes_length)
            <= max_time_series_time
    ):
        current_time_frame_end = (
                current_time_frame_start
                + datetime.timedelta(minutes=time_frame_minutes_length)
                - datetime.timedelta(seconds=1)
        )
        time_frames.append((current_time_frame_start, current_time_frame_end))
        current_time_frame_start += datetime.timedelta(minutes=time_frame_minutes_length)
    df_time_frames = pd.DataFrame(
        data=time_frames, columns=["start", "end"]
    )
    df_time_frames["start_hour"] = df_time_frames.start.dt.time
    df_time_frames["end_hour"] = df_time_frames.end.dt.time
    df_time_frames[LABEL_COLUMN_NAME] = df_time_frames["start_hour"].astype(str) + " - " + df_time_frames[
        "end_hour"].astype(str)
    return df_time_frames


def _prepare_figure(
        df: pd.DataFrame,
        time_frames,
        y_label: str,
        file_name: str,
        is_stacked_bar_plot: bool = False
):
    if is_stacked_bar_plot:
        plot = df.plot.bar(stacked=True)
    else:
        plot = df.plot()
    plot.set_xticks(time_frames.index)
    plot.set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    plot.set_xlabel("Time frame", fontsize=15, labelpad=15)
    plot.set_ylabel(y_label, fontsize=15, labelpad=15)
    plot.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        fancybox=True,
        shadow=True,
        ncol=4,
        # labels=["negative", "neutral", "positive"]
    )
    plt.tight_layout()
    plot.figure.savefig(file_name)
    plt.show()


def _show_topic_statistics(df_topic: pd.DataFrame, time_frames, total_tweets_per_time_frame):
    df = (
        df_topic
            .groupby(["time_frame", "topic"])["tweet"]
            .count()
            .reset_index(name="count")
    )
    df = df.merge(total_tweets_per_time_frame, on="time_frame")
    df["percentage"] = df["count"] / df["time_frame_count"] * 100
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    topic_count_series = {}
    topic_percentage_series = {}
    for topic in df["topic"].unique():
        topic_name = f"Topic {int(topic) + 1}"
        topic_count_series[topic_name] = df[df.topic == topic].set_index("time_frame")["count"]
        topic_percentage_series[topic_name] = df[df.topic == topic].set_index("time_frame")["percentage"]

    df_topic_count_series = pd.DataFrame(topic_count_series)
    _prepare_figure(
        df=df_topic_count_series,
        time_frames=time_frames,
        y_label="Tweet count",
        file_name="results/topics_absolute_linear",
    )
    df_topic_percentage_series = pd.DataFrame(topic_percentage_series)
    _prepare_figure(
        df=df_topic_percentage_series,
        time_frames=time_frames,
        y_label="Tweet percentage [%]",
        file_name="results/topics_percentage_linear",
    )
    _prepare_figure(
        df=df_topic_count_series,
        time_frames=time_frames,
        y_label="Tweet count",
        file_name="results/topics_absolute_bar",
        is_stacked_bar_plot=True,
    )
    _prepare_figure(
        df=df_topic_percentage_series,
        time_frames=time_frames,
        y_label="Tweet percentage [%]",
        file_name="results/topics_percentage_bar",
        is_stacked_bar_plot=True,
    )

    print(df)


def _show_sentiment_statistics(df_sentiment: pd.DataFrame, time_frames, total_tweets_per_time_frame):
    df = (
        df_sentiment
            .groupby(["time_frame", LABEL_COLUMN_NAME])[TEXT_COLUMN_NAME]
            .count()
            .reset_index(name="count")
    )
    df = df.merge(total_tweets_per_time_frame, on="time_frame")
    df["percentage"] = df["count"] / df["time_frame_count"] * 100
    sentiment_count_series = {}
    sentiment_percentage_series = {}
    for sentiment in df[LABEL_COLUMN_NAME].unique():
        sentiment_count_series[sentiment] = df[df[LABEL_COLUMN_NAME] == sentiment].set_index("time_frame")["count"]
        sentiment_percentage_series[sentiment] = df[df[LABEL_COLUMN_NAME] == sentiment].set_index("time_frame")[
            "percentage"]

    df_sentiment_count_series = pd.DataFrame(sentiment_count_series)
    df_sentiment_percentage_series = pd.DataFrame(sentiment_percentage_series)
    _prepare_figure(
        df=df_sentiment_count_series,
        time_frames=time_frames,
        y_label="Tweet count",
        file_name="results/sentiment_absolute_linear",
    )
    _prepare_figure(
        df=df_sentiment_percentage_series,
        time_frames=time_frames,
        y_label="Tweet percentage [%]",
        file_name="results/sentiment_percentage_linear",
    )
    _prepare_figure(
        df=df_sentiment_count_series,
        time_frames=time_frames,
        y_label="Tweet count",
        file_name="results/sentiment_absolute_bar",
        is_stacked_bar_plot=True,
    )
    _prepare_figure(
        df=df_sentiment_percentage_series,
        time_frames=time_frames,
        y_label="Tweet percentage [%]",
        file_name="results/sentiment_percentage_bar",
        is_stacked_bar_plot=True,
    )
    print(df)


@section_printing_decorator
def show_time_frames_analysis(
        df_sentiment: pd.DataFrame,
        df_topic: pd.DataFrame,
        time_frame_minutes_length: int = 30
):
    print("7. TIME FRAME SENTIMENT AND TOPICS ANALYSIS")
    print()
    sentiment_time_series = df_sentiment[CREATED_AT_COLUMN_NAME].apply(pd.to_datetime)
    time_frames = _get_time_frames(
        time_series=sentiment_time_series,
        time_frame_minutes_length=time_frame_minutes_length,
    )

    # sentiment
    sentiment_time_frame_labels = sentiment_time_series.apply(
        lambda time_record: time_frames[
            (time_record >= time_frames["start"])
            & (time_record <= time_frames["end"])
            ].index[0]
    )
    df_sentiment["time_frame"] = sentiment_time_frame_labels
    labeled_df_vader = Labeler.get_data_sentiment_with_vader(
        df=df_sentiment, polarity_absolute_threshold=0.05,
    )
    df_sentiment[LABEL_COLUMN_NAME] = labeled_df_vader[LABEL_COLUMN_NAME]

    # topic
    topic_time_series = df_topic[CREATED_AT_COLUMN_NAME].apply(pd.to_datetime)
    topic_time_frame_labels = topic_time_series.apply(
        lambda time_record: time_frames[
            (time_record >= time_frames["start"])
            & (time_record <= time_frames["end"])
            ].index[0]
    )
    df_topic["time_frame"] = topic_time_frame_labels
    total_tweets_per_time_frame = (
        df_topic
        .groupby('time_frame')["tweet"]
        .count()
        .reset_index(name="time_frame_count")
    )
    figure = total_tweets_per_time_frame.time_frame_count.plot()
    figure.set_xticks(time_frames.index)
    figure.set_xlabel("Time frame", fontsize=15, labelpad=15)
    figure.set_ylabel("Tweet count", fontsize=15, labelpad=15)
    figure.set_xticklabels(time_frames[LABEL_COLUMN_NAME], rotation=90)
    figure.figure.savefig("results/total_tweets_time_frame_real_liv.png")
    plt.show()

    _show_topic_statistics(
        df_topic=df_topic,
        total_tweets_per_time_frame=total_tweets_per_time_frame,
        time_frames=time_frames,
    )
    # _show_sentiment_statistics(
    #     df_sentiment=df_sentiment,
    #     total_tweets_per_time_frame=total_tweets_per_time_frame,
    #     time_frames=time_frames,
    # )
    # print(df_sentiment)
    # print(time_frames)

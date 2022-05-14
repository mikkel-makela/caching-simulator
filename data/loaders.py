from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


MAX_TRACE_LENGTH: int or None = 50_000
MAX_ROWS: int or None = 1_000_000


@dataclass
class Dataset:
    trace: np.ndarray
    catalog_size: int


class MovielensColumns:
    movie_id: str = "movieId"
    timestamp: str = "timestamp"
    user_id: str = "userId"


def load_movielens(file_path: str) -> Dataset:
    df = pd.read_csv(
        file_path,
        usecols=[MovielensColumns.timestamp, MovielensColumns.movie_id]
    )
    df = df[df[MovielensColumns.movie_id] < MAX_TRACE_LENGTH].head(MAX_TRACE_LENGTH)
    df.sort_values(by=MovielensColumns.timestamp, ascending=True)
    return Dataset(
        catalog_size=df[MovielensColumns.movie_id].max(),
        trace=df[MovielensColumns.movie_id].to_numpy()
    )


def load_bipartite_movielens(file_path: str) -> np.ndarray:
    """
    Returns an array of datasets, each corresponding to one user.
    :param file_path: path to raw movielens data file
    :return: numpy array of Datasets
    """

    df = pd.read_csv(
        file_path,
        usecols=[
            MovielensColumns.timestamp,
            MovielensColumns.movie_id,
            MovielensColumns.user_id
        ],
        nrows=MAX_ROWS
    )
    df = df[df[MovielensColumns.movie_id] < MAX_TRACE_LENGTH]\
        .head(MAX_TRACE_LENGTH)\
        .groupby([MovielensColumns.user_id])

    user_ids: List[int] = [user_id for user_id in df.groups.keys()]
    datasets: List[Dataset] = list(
        map(
            lambda user_df: Dataset(
                catalog_size=user_df[MovielensColumns.movie_id].max(),
                trace=user_df[MovielensColumns.movie_id].to_numpy()
            ),
            map(lambda user_id: df.get_group(name=user_id), user_ids)
        )
    )

    return np.array(datasets)

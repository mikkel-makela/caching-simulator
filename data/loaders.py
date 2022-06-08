import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


MAX_TRACE_LENGTH: int or None = 5_000
MAX_ROWS: int or None = 1_000_000


@dataclass
class Dataset:
    trace: np.ndarray
    catalog_size: int
    name: str


class MovielensColumns:
    movie_id: str = "movieId"
    timestamp: str = "timestamp"
    user_id: str = "userId"


def load_movielens(file_path: str, cache_size: int, trace_length: int = MAX_TRACE_LENGTH) -> Dataset:
    df = pd.read_csv(
        file_path,
        usecols=[MovielensColumns.timestamp, MovielensColumns.movie_id]
    )
    df = df[df[MovielensColumns.movie_id] < cache_size * 100].head(trace_length)
    df.sort_values(by=MovielensColumns.timestamp, ascending=True)
    assert df.shape[0] == trace_length, "Too many movies filtered out, provided trace length was not achieved."
    return Dataset(
        catalog_size=df[MovielensColumns.movie_id].max(),
        trace=df[MovielensColumns.movie_id].to_numpy(),
        name=f'MovieLens {trace_length}'
    )


def load_bipartite_movielens(file_path: str, cache_size: int, trace_length: int = MAX_TRACE_LENGTH) -> np.ndarray:
    """
    Returns an array of datasets, each corresponding to one user.
    :param cache_size: The size of the cache this dataset will be used on
    :param file_path: path to raw movielens data file
    :param trace_length: How many requests to include
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
    df = df[df[MovielensColumns.movie_id] < cache_size * 100]\
        .head(trace_length)\
        .groupby([MovielensColumns.user_id])

    user_ids: List[int] = [user_id for user_id in df.groups.keys()]
    datasets: List[Dataset] = list(
        map(
            lambda user_df: Dataset(
                catalog_size=user_df[MovielensColumns.movie_id].max(),
                trace=user_df[MovielensColumns.movie_id].to_numpy(),
                name="movielens"
            ),
            map(lambda user_id: df.get_group(name=user_id), user_ids)
        )
    )

    return np.array(datasets)


def load_online_cache_trace(file_name: str) -> Dataset:
    with open(file_name, 'rb') as f:
        trace = pickle.load(f)
        return Dataset(
            catalog_size=np.max(trace),
            trace=trace,
            name=file_name
        )

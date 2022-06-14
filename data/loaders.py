import pickle
from dataclasses import dataclass
from typing import List


import numpy as np
import pandas as pd

from data.path import DataPath


@dataclass
class AbstractDataset:
    catalog_size: int
    name: str


@dataclass
class Dataset(AbstractDataset):
    """
    Contains a variable trace, which is a 1 x T array
    """
    trace: np.ndarray


@dataclass
class BiPartiteDataset(AbstractDataset):
    """
    Contains a variable traces, which is a clients x T array
    """
    traces: np.ndarray


class MovielensColumns:
    movie_id: str = "movieId"
    timestamp: str = "timestamp"
    user_id: str = "userId"
    id: str = "id"


def with_compressed_ids(df: pd.DataFrame):
    df[MovielensColumns.id] = pd.factorize(df[MovielensColumns.movie_id])[0]
    return df


def load_movielens(file_path: str, catalog_size: int, trace_length: int) -> Dataset:
    df = with_compressed_ids(
        pd.read_csv(
            file_path,
            usecols=[MovielensColumns.timestamp, MovielensColumns.movie_id]
        )
    )
    df = df[df[MovielensColumns.id] < catalog_size].head(trace_length)
    df.sort_values(by=MovielensColumns.timestamp, ascending=True)
    assert df.shape[0] == trace_length, "Too many movies filtered out, provided trace length was not achieved."
    return Dataset(
        catalog_size=df[MovielensColumns.id].max(),
        trace=df[MovielensColumns.id].to_numpy(),
        name=f'MovieLens {trace_length}'
    )


def load_online_cache_trace(file_name: str) -> Dataset:
    with open(file_name, 'rb') as f:
        trace = pickle.load(f)
        return Dataset(
            catalog_size=np.max(trace) + 1,
            trace=trace,
            name=file_name
        )


def load_bipartite_traces() -> BiPartiteDataset:
    synthetic_datasets: List[Dataset] = list(map(
        lambda file_name: load_online_cache_trace(file_name),
        [
            DataPath.OSCILLATOR,
            DataPath.CHANGING_OSCILLATOR,
            DataPath.CHANGING_POPULARITY_CATALOG,
            DataPath.FIXED_POPULARITY_CATALOG,
            DataPath.SN_OSCILLATOR
        ]
    ))
    catalog_size = min(synthetic_datasets, key=lambda ds: ds.catalog_size).catalog_size
    for dataset in synthetic_datasets:
        dataset.trace = dataset.trace[dataset.trace <= catalog_size]
    time_horizon = min(synthetic_datasets, key=lambda ds: ds.trace.size).trace.size
    for dataset in synthetic_datasets:
        dataset.trace = dataset.trace[:time_horizon]
    all_traces = np.array(
        list(map(lambda ds: ds.trace, synthetic_datasets)) + [
            load_movielens(DataPath.MOVIE_LENS, catalog_size, time_horizon).trace
        ]
    )
    return BiPartiteDataset(
        name="Synthetic + MovieLens",
        catalog_size=catalog_size + 1,
        traces=all_traces
    )

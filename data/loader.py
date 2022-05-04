from dataclasses import dataclass

import numpy as np
import pandas as pd


TRACE_LENGTH: int or None = 10_000


@dataclass
class Dataset:
    trace: np.ndarray
    catalog_size: int


class MovielensColumns:
    movie_id: str = "movieId"
    timestamp: str = "timestamp"


def load_movie_lens(file_path: str) -> Dataset:
    df = pd.read_csv(file_path, usecols=[MovielensColumns.timestamp, MovielensColumns.movie_id], nrows=TRACE_LENGTH)
    df.sort_values(by=MovielensColumns.timestamp, ascending=True)
    return Dataset(
        catalog_size=df[MovielensColumns.movie_id].max(),
        trace=df[MovielensColumns.movie_id].to_numpy()
    )

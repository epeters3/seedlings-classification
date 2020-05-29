import os
from typing import NamedTuple
import pandas as pd


class RunResult(NamedTuple):
    dev_acc: float
    train_acc: float
    run_name: str
    submission: pd.DataFrame


def make_dir_if_needed(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_sub_dirs(dir_path: str) -> list:
    return [f.path for f in os.scandir(dir_path) if f.is_dir()]


def get_dir_files(dir_path: str) -> list:
    """
    Gets the names of the files that are direct children of `dir_path`.
    """
    return [
        fname
        for fname in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, fname))
    ]

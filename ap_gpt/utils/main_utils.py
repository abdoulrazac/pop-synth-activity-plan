import os
import sys
from typing import Union

import dill
import numpy as np
import pandas as pd
import torch
import yaml

from ap_gpt.ap_exception import APException
from ap_gpt.ap_logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise APException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise APException(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise APException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise APException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise APException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise APException(e, sys) from e


def read_data(file_path: str, index_col=None, is_array:bool=False) -> Union[pd.DataFrame, np.ndarray]:
    """
    Reads data from a file and returns it as a DataFrame.

    Args:
        file_path (str): Path to the file.
        index_col (int, optional): Column to set as index. Defaults to None.
        is_array (bool, optional): Whether to return a numpy array. Defaults to False.
    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    try:
        extension = os.path.splitext(file_path)[1]

        if is_array or extension == ".npy":
            return np.load(file_path, allow_pickle=True)
        elif extension == ".csv" or extension == ".txt":
            return pd.read_csv(file_path, index_col=index_col)
        elif extension == ".json":
            return pd.read_json(file_path)
        elif extension == ".xls" or extension == ".xlsx":
            return pd.read_excel(file_path, index_col=index_col)
        elif extension == ".parquet":
            return pd.read_parquet(file_path)
        elif extension == ".feather":
            return pd.read_feather(file_path)
        else:
            raise ValueError("Unknown format")

    except Exception as e:
        raise APException(e, sys)


def save_data(data: Union[pd.DataFrame, np.ndarray], file_path: str, index: bool = False) -> None:
    """
    Saves a DataFrame to a file.

    Args:
        data (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the file.
        index (bool, optional): Whether to save the index. Defaults to False.
    """
    try:
        extension = os.path.splitext(file_path)[1]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if isinstance(data, np.ndarray) or extension == ".npy":
            np.save(file_path, data, allow_pickle=True)
        elif extension == ".csv" or extension == ".txt":
            data.to_csv(file_path, index=index)
        elif extension == ".json":
            data.to_json(file_path, orient="records", lines=True)
        elif extension == ".xls" or extension == ".xlsx":
            data.to_excel(file_path, index=index)
        elif extension == ".parquet":
            data.to_parquet(file_path, index=index)
        elif extension == ".feather":
            data.to_feather(file_path)
        else:
            raise ValueError("Unknown format")

    except Exception as e:
        raise APException(e, sys)


def split_data(data: pd.DataFrame, train_size : float = None, test_size: float = None, random_state=123) -> tuple:
    """
    Splits the data into training and testing sets.

    Args:
        data (pd.DataFrame): DataFrame to split.
        train_size (float): Proportion of the dataset to include in the train split.
        test_size (float): Proportion of the dataset to include in the test split. Defaults to None.
        random_state (int, optional): Random seed for reproducibility. Defaults to 123.

    Returns:
        tuple: Training and testing DataFrames.
    """
    try:
        if train_size is None and test_size is None:
            raise ValueError("Either train_size or test_size must be provided.")
        train_data = data.sample(frac=1 - test_size if not test_size is None else train_size , random_state=random_state)
        test_data = data.drop(train_data.index)
        return train_data, test_data

    except Exception as e:
        raise APException(e, sys)


def get_device() -> torch.device:
    """
    Get the device type (CPU or GPU) based on availability.

    Returns:
        str: Device type.
    """

    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     return torch.device("mps")
        else:
            return torch.device("cpu")
    except Exception as e:
        raise APException(e, sys)

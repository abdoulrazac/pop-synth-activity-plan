import sys
from abc import ABC, abstractmethod
from typing import Optional, List, Union

import pandas as pd

from ap_gpt.ap_exception import APException
from ap_gpt.constants import *
from ap_gpt.utils.main_utils import read_yaml_file


class DataProcessingBase(ABC):
    def __init__(self) -> None:
        try:
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise APException(e, sys)

    def is_valid_data(self, data: pd.DataFrame, table_name: str, other_columns : Union[List, None] = None) -> Optional[bool]:
        """
        Method Name :   is_valid_data
        Description :   This method checks if the data is valid

        Args:
            data (pd.DataFrame): Dataframe to check
            table_name (str): Name of the table to check
            other_columns (list): List of other columns to check

        Returns:
            bool: True if the data is valid, False otherwise
        """

        if other_columns is None:
            other_columns = []
        try:
            # check if identifier column is present
            if self._schema_config[table_name][SCHEMA_IDENTIFIER_NAME] not in data.columns:
                raise f"Identifier column not present in data: {table_name}"

            # check if weight column is present
            if self._schema_config[table_name][SCHEMA_WEIGHT_NAME] not in data.columns:
                raise f"Weight column not present in data: {table_name}"

            # check if numerical columns are present
            for column in self._schema_config[table_name][SCHEMA_NUMERICAL_NAME]:
                if column not in data.columns:
                    raise f"Numerical column named ''{column}'' is not present in data: {column}"

            # check if categorical columns are present
            for column in self._schema_config[table_name][SCHEMA_CATEGORICAL_NAME]:
                if column not in data.columns:
                    raise f"Categorical column named ''{column}'' is not present in data: {column}"

            # check if other columns are present
            for column in other_columns:
                if column not in data.columns:
                    raise f"Column named ''{column}'' is not present in data: {column}"

        except Exception as e:
            raise APException(e, sys)

    def keep_required_columns(self, data: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Method Name :   keep_required_columns
        Description :   This method keeps only the required columns in the dataframe

        Args:
            data (pd.DataFrame): Dataframe to keep columns
            table_name (str): Name of the table to keep columns

        Returns:
            pd.DataFrame: Dataframe with only the required columns
        """

        try:
            columns = [
                self._schema_config[table_name][SCHEMA_IDENTIFIER_NAME],
                self._schema_config[table_name][SCHEMA_WEIGHT_NAME],
                *self._schema_config[table_name][SCHEMA_NUMERICAL_NAME],
                *self._schema_config[table_name][SCHEMA_CATEGORICAL_NAME],
            ]

            return data[columns]
        except Exception as e:
            raise APException(e, sys)

    def recode_numerical_columns(self, data: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Recode numerical columns

        Args:
            data (pd.DataFrame): Dataframe to recode
            table_name (str): Name of the table to recode

        Returns:
            pd.DataFrame: Dataframe with recoded columns
        """
        recode_info = self._schema_config[table_name][SCHEMA_NUMERICAL_NAME]
        new_dat = data.copy()
        for col in recode_info.keys():
            tmp = np.array(data[col].values)
            for k, v in recode_info[col].items():
                a = tmp >= v[0]
                b = tmp < v[1]
                new_dat.loc[a & b, col] = k
            new_dat[col] = new_dat[col].astype(object)
        return new_dat

    def recode_categorical_columns(self, data: pd.DataFrame, table_name) -> pd.DataFrame:
        """Recode categorical columns

        Args:
            data (pd.DataFrame): Dataframe to recode
            table_name (str): Name of the table to recode

        Returns:
            pd.DataFrame: Dataframe with recoded columns
        """

        recode_info = self._schema_config[table_name][SCHEMA_CATEGORICAL_NAME]
        new_dat = data.copy()
        for col in recode_info.keys():
            for k, v in recode_info[col].items():
                new_dat.loc[data[col].isin(v), col] = k
            new_dat[col] = new_dat[col].astype(object)
        return new_dat

    @abstractmethod
    def initiate_data_processing(self) :
        pass

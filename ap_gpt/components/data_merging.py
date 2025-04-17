import sys
import os
import pandas as pd
from from_root import from_root

from ap_gpt.constants import SCHEMA_FILE_PATH, TABLE_PERSON_NAME, SCHEMA_IDENTIFIER_NAME, TABLE_HOUSEHOLD_NAME, \
    TABLE_TRIP_NAME, SCHEMA_WEIGHT_NAME
from ap_gpt.entity.artifact_entity import (HouseholdDataProcessingArtifact, PersonDataProcessingArtifact,
                                           TripDataProcessingArtifact, DataMergingArtifact, DataProcessingArtifact)
from ap_gpt.entity.config_entity import DataMergingConfig
from ap_gpt.ap_exception import APException
from ap_gpt.utils.main_utils import read_data, read_yaml_file, save_data
from ap_gpt.ap_logger import logging

from ap_gpt.utils  import value_prefixer


class DataMerging:
    def __init__(self,
                 data_processing_artifact: DataProcessingArtifact,
                 data_merging_config: DataMergingConfig = DataMergingConfig(),
                 ):
        """
        Args:
            data_processing_artifact (HouseholdDataProcessingArtifact): ata processing artifact.
            data_merging_config (DataMergingConfig, optional): Data merging configuration. Defaults to DataMergingConfig().
        """

        self.data_processing_artifact = data_processing_artifact
        self.data_merging_config = data_merging_config
        self._schema_config = read_yaml_file(file_path=os.path.join(from_root(), SCHEMA_FILE_PATH))

        self.household_id_col = self._schema_config[TABLE_HOUSEHOLD_NAME][SCHEMA_IDENTIFIER_NAME]
        self.person_id_col = self._schema_config[TABLE_PERSON_NAME][SCHEMA_IDENTIFIER_NAME]
        self.trip_id_col = self._schema_config[TABLE_TRIP_NAME][SCHEMA_IDENTIFIER_NAME]
        self.household_weight_col = self._schema_config[TABLE_HOUSEHOLD_NAME][SCHEMA_WEIGHT_NAME]
        self.person_weight_col = self._schema_config[TABLE_PERSON_NAME][SCHEMA_WEIGHT_NAME]
        self.trip_weight_col = self._schema_config[TABLE_TRIP_NAME][SCHEMA_WEIGHT_NAME]



    def merge_all_data(self, household, person, trip) -> pd.DataFrame:
        """
        Merges household, person, and trip data into a single DataFrame.

        Args:
            household (pd.DataFrame): Household data.
            person (pd.DataFrame): Person data.
            trip (pd.DataFrame): Trip data.

        Returns:
            pd.DataFrame: Merged DataFrame containing household, person, and trip data.
        """
        try:
            # Merge the dataframes
            df_merged = household.merge(person, on=self.household_id_col, how="right").merge(
                trip, on=self.person_id_col, how="right"
            )
            return df_merged
        except Exception as e:
            raise APException(e, sys)

    @staticmethod
    def weight_data(data: pd.DataFrame, weight_col : str) -> pd.DataFrame:
        """
        Applies data weighting to the merged DataFrame.

        Args:
            data (pd.DataFrame): Merged DataFrame.
            weight_col (str): Column name for weighting.

        Returns:
            pd.DataFrame: DataFrame with applied data weighting.
        """
        try:
            return data.reindex(data.index.repeat(data[weight_col])).reset_index(drop=True)
        except Exception as e:
            raise APException(e, sys)


    def remove_id_and_weight_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes ID and weight columns from the DataFrame.

        Args:
            data (pd.DataFrame): DataFrame to remove columns from.

        Returns:
            pd.DataFrame: DataFrame with ID and weight columns removed.
        """
        try:
            # Remove ID and weight columns if they exist
            id_cols = [self.household_id_col, self.person_id_col, self.trip_id_col]
            weight_cols = [self.household_weight_col, self.person_weight_col, self.trip_weight_col]
            cols_to_remove = [col for col in (id_cols + weight_cols) if col in data.columns]
            data.drop(columns=cols_to_remove, inplace=True, errors='ignore')

            return data
        except Exception as e:
            raise APException(e, sys)


    def initiate_data_merging(self) -> DataMergingArtifact:
        """
        Merges household, person, and trip data into a single DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame containing household, person, and trip data.
        """
        try:
            # Read the processed data files
            logging.info("Reading processed data files")
            df_household = read_data(self.data_processing_artifact.household_processed_data_file_path)
            df_person = read_data(self.data_processing_artifact.person_processed_data_file_path)
            df_trip = read_data(self.data_processing_artifact.trip_processed_data_file_path)

            # Add Value prefix
            logging.info("Adding Value prefix to household data")
            df_household = value_prefixer.transform_all(df_household, exclude=[self.household_id_col, self.household_weight_col])
            df_person = value_prefixer.transform_all(df_person, exclude=[self.household_id_col, self.person_id_col, self.person_weight_col])

            # Merge the dataframes
            logging.info("Merging household, person, and trip data")
            df_merged = self.merge_all_data(df_household, df_person, df_trip)
            logging.info(f"Shape of merged data: {df_merged.shape}")

            # # Apply data weighting
            # logging.info("Applying data weighting to merged data using person weight")
            # df_merged = self.weight_data(df_merged, self.person_weight_col)
            # logging.info(f"Shape of weighted merged data: {df_merged.shape}")

            # Remove unnecessary columns
            logging.info("Removing ID and weight columns from merged data")
            df_merged = self.remove_id_and_weight_columns(df_merged)

            data_merging_artifact = DataMergingArtifact(
                merged_data_file_path=self.data_merging_config.merged_data_file_path,
            )

            # # Save the merged dataframe
            logging.info("Saving merged data")
            save_data(df_merged, self.data_merging_config.merged_data_file_path)

            return data_merging_artifact
        except Exception as e:
            raise APException(e, sys)

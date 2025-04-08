import sys
import pandas as pd

from ap_gpt.constants import SCHEMA_FILE_PATH, TABLE_PERSON_NAME, SCHEMA_IDENTIFIER_NAME, TABLE_HOUSEHOLD_NAME
from ap_gpt.entity.artifact_entity import (HouseholdDataProcessingArtifact, PersonDataProcessingArtifact,
                                           TripDataProcessingArtifact, DataMergingArtifact)
from ap_gpt.entity.config_entity import DataMergingConfig
from ap_gpt.ap_exception import APException
from ap_gpt.utils.main_utils import read_data, read_yaml_file, save_data
from ap_gpt.ap_logger import logging
from ap_gpt.utils.value_prefixer import ValuePrefixer


class DataMerging:
    def __init__(self,
                 household_data_processing_artifact: HouseholdDataProcessingArtifact,
                 person_data_processing_artifact: PersonDataProcessingArtifact,
                 trip_data_processing_artifact: TripDataProcessingArtifact,
                 data_merging_config: DataMergingConfig = DataMergingConfig(),
                 ):
        """
        Args:
            household_data_processing_artifact (HouseholdDataProcessingArtifact): Household data processing artifact.
            person_data_processing_artifact (PersonDataProcessingArtifact): Person data processing artifact.
            trip_data_processing_artifact (TripDataProcessingArtifact): Trip data processing artifact.
            data_merging_config (DataMergingConfig, optional): Data merging configuration. Defaults to DataMergingConfig().
        """

        self.household_data_processing_artifact = household_data_processing_artifact
        self.person_data_processing_artifact = person_data_processing_artifact
        self.trip_data_processing_artifact = trip_data_processing_artifact
        self.data_merging_config = data_merging_config
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    def initiate_data_merging(self) -> DataMergingArtifact:
        """
        Merges household, person, and trip data into a single DataFrame.

        Returns:
            pd.DataFrame: Merged DataFrame containing household, person, and trip data.
        """
        try:

            # Read the schema configuration
            household_id_col = self._schema_config[TABLE_HOUSEHOLD_NAME][SCHEMA_IDENTIFIER_NAME]
            person_id_col = self._schema_config[TABLE_PERSON_NAME][SCHEMA_IDENTIFIER_NAME]

            # Read the processed data files
            logging.info("Reading processed data files")
            df_household = read_data(self.household_data_processing_artifact.household_processed_data_file_path)
            df_person = read_data(self.person_data_processing_artifact.person_processed_data_file_path)
            df_trip = read_data(self.trip_data_processing_artifact.trip_processed_data_file_path)

            # Add Value prefix
            logging.info("Adding Value prefix to household data")
            df_household = ValuePrefixer.transform_all(df_household, exclude=[household_id_col])
            df_person = ValuePrefixer.transform_all(df_person, exclude=[household_id_col, person_id_col])

            # Merge the dataframes
            logging.info("Merging household, person, and trip data")
            df_merged = df_trip.merge(df_person, on=person_id_col, how="left").merge(
                df_household, on=household_id_col, how="left"
            )

            data_merging_artifact = DataMergingArtifact(
                merged_data_file_path=self.data_merging_config.merged_data_file_path,
            )

            # Save the merged dataframe
            logging.info("Saving merged data")
            save_data(df_merged, data_merging_artifact.merged_data_file_path)

            return data_merging_artifact
        except Exception as e:
            raise APException(e, sys)

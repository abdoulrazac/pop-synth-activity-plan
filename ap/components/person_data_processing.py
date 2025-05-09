import sys
import pandas as pd

from typing import Optional

from ap.ap_exception import APException
from ap.ap_logger import logging
from ap.components.data_processing_base import DataProcessingBase
from ap.constants import *
from ap.entity.artifact_entity import DataIngestionArtifact, PersonDataProcessingArtifact
from ap.entity.config_entity import DataProcessingConfig
from ap.utils.main_utils import read_data, save_data


class PersonDataProcessing(DataProcessingBase):
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_processing_config: DataProcessingConfig = DataProcessingConfig(),
                 ) -> None:
        """
        :param data_ingestion_artifact:
        :param data_processing_config:
        """
        try:
            super().__init__()
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_processing_config = data_processing_config
            self.table_name = TABLE_PERSON_NAME
            self.max_trip_number = MAX_TRIP_NUMBER
            self.number_of_trips_col =  TABLE_PERSON_NUMBER_OF_TRIPS_NAME
        except Exception as e:
            raise APException(e, sys)

    # Function to keep only interest row
    def keep_required_row(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method of DataProcessing class is responsible for keeping only the required rows
        """
        try:
            # Keep only the required rows
            return data[(data[self.number_of_trips_col] >= 0) & (data[self.number_of_trips_col] <= self.max_trip_number)]
        except Exception as e:
            raise APException(e, sys)

    def initiate_data_processing(self) -> Optional[PersonDataProcessingArtifact]:
        """
        Method Name :   initiate_data_processing
        Description :   This method initiates the data processing component for the pipeline
        """

        try :
            logging.info("Entered person initiate_data_processing method of DataProcessing class")

            # Read the data from the CSV files
            logging.info("Reading person data from CSV files")
            df_person = read_data(self.data_ingestion_artifact.person_data_file_path)

            # Make filter on the data
            logging.info("Filtering person data")
            df_person = self.keep_required_row(df_person)

            # Keep only the required columns
            logging.info("Keeping only the person's required columns")
            df_person = self.keep_required_columns(df_person, self.table_name)

            # Validate the data
            logging.info("Validating person data")
            is_valid_data = self.is_valid_data(df_person, self.table_name)
            if not is_valid_data:
                raise ValueError(f"Invalid data in person data: {self.table_name}")

            # Remove unnecessary col
            logging.info("Remove unnecessary columns from person data")
            df_person = self.remove_unnecessary_columns(df_person, self.table_name)

            # recode numerical columns
            logging.info("Recode person numerical columns")
            df_person = self.cut_numerical_columns(df_person, self.table_name)

            logging.info("Create DataProcessing artifact")
            data_processing_artifact = PersonDataProcessingArtifact(
                person_processed_data_file_path = self.data_processing_config.person_processed_data_file_path,
            )

            # Save person data
            logging.info("Saving person data")
            logging.info(f"Shape of person data after processing : {df_person.shape}")
            save_data(df_person, data_processing_artifact.person_processed_data_file_path)

            return data_processing_artifact

        except Exception as e :
            APException(e, sys)

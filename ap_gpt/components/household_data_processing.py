import sys
from typing import Optional

from ap_gpt.ap_exception import APException
from ap_gpt.ap_logger import logging
from ap_gpt.components.data_processing_base import DataProcessingBase
from ap_gpt.constants import *
from ap_gpt.entity.artifact_entity import DataIngestionArtifact, HouseholdDataProcessingArtifact
from ap_gpt.entity.config_entity import DataProcessingConfig
from ap_gpt.utils.main_utils import read_data, save_data


class HouseholdDataProcessing(DataProcessingBase):
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
            self.table_name = TABLE_HOUSEHOLD_NAME
        except Exception as e:
            raise APException(e, sys)

    def initiate_data_processing(self) -> Optional[HouseholdDataProcessingArtifact]:
        """
        Method Name :   initiate_data_processing
        Description :   This method initiates the data processing component for the pipeline
        """

        try :
            logging.info("Entered household initiate_data_processing method of DataProcessing class")

            # Read the data from the CSV files
            logging.info("Reading data from CSV files")
            df_household = read_data(self.data_ingestion_artifact.household_data_file_path)

            # Validate the data
            logging.info("Validating household data")
            is_valid_data = self.is_valid_data(df_household, self.table_name)
            if not is_valid_data:
                raise ValueError(f"Invalid data in household data: {self.table_name}")

            # Keep only the required columns
            logging.info("Keeping only the household's required columns")
            df_household = self.keep_required_columns(df_household, self.table_name)

            # recode numerical columns
            logging.info("Recode household numerical columns")
            df_household = self.recode_numerical_columns(df_household, self.table_name)

            logging.info("Create household DataProcessing artifact")
            data_processing_artifact: HouseholdDataProcessingArtifact = HouseholdDataProcessingArtifact(
                household_processed_data_file_path = self.data_processing_config.household_processed_data_file_path,
            )

            # Save household data
            logging.info("Saving household data")
            save_data(df_household, data_processing_artifact.household_processed_data_file_path)

            return data_processing_artifact

        except Exception as e :
            APException(e, sys)

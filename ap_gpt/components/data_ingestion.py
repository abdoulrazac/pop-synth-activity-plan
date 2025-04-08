import os
import sys
from dataclasses import asdict
from typing import Tuple

import pandas as pd

from ap_gpt.ap_exception import APException
from ap_gpt.entity.artifact_entity import DataIngestionArtifact
from ap_gpt.entity.config_entity import DataIngestionConfig
from ap_gpt.ap_logger import logging
from ap_gpt.utils.main_utils import read_data, save_data


class DataIngestion :
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise APException(e, sys)


    def export_data_into_feature_store(self)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        """
        try:
            logging.info("Exporting data to feature store")
            df_household = read_data(self.data_ingestion_config.household_raw_data_file_path, index_col=0)
            df_person = read_data(self.data_ingestion_config.person_raw_data_file_path, index_col=0)
            df_trips = read_data(self.data_ingestion_config.trip_raw_data_file_path, index_col=0)

            logging.info(f"Shape of household data: {df_household.shape}")
            logging.info(f"Shape of person data: {df_person.shape}")
            logging.info(f"Shape of trips data: {df_trips.shape}")

            # Save the dataframes to CSV files
            os.makedirs(self.data_ingestion_config.data_store_path, exist_ok=True)

            # Save dataframes
            logging.info(f"Saving all dataframes")
            save_data(df_household,  os.path.join(self.data_ingestion_config.household_data_file_path))
            save_data(df_person, os.path.join(self.data_ingestion_config.person_data_file_path))
            save_data(df_trips, os.path.join(self.data_ingestion_config.trip_data_file_path))

            return df_household, df_person, df_trips

        except Exception as e:
            raise APException(e, sys)


    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion component for the pipeline
        """

        logging.info("Initiating data ingestion")

        try :
            logging.info("Get data")
            self.export_data_into_feature_store()

            logging.info("Create artifact")
            data_ingestion_artifact = DataIngestionArtifact(
                household_data_file_path=self.data_ingestion_config.household_data_file_path,
                person_data_file_path=self.data_ingestion_config.person_data_file_path,
                trip_data_file_path=self.data_ingestion_config.trip_data_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise APException(e, sys)

import sys

from ap.entity.artifact_entity import DataMergingArtifact, DataSplittingArtifact
from ap.entity.config_entity import DataSplittingConfig
from ap.utils.main_utils import read_data, split_data, save_data
from ap.ap_logger import logging
from ap.ap_exception import APException


class DataSplitting :
    def __init__(self,
                 data_merging_artifact : DataMergingArtifact,
                 data_splitting_config : DataSplittingConfig = DataSplittingConfig()
                 ) -> None:

        self.data_merging_artifact = data_merging_artifact
        self.data_splitting_config = data_splitting_config

    def initiate_data_splitting(self) -> DataSplittingArtifact:
        """
        Splits the merged data into training and testing sets.

        Returns:
            DataSplittingArtifact: Artifact containing file paths for training and testing data.
        """

        try:
            # Read the merged data
            logging.info("Initiating data splitting")
            df = read_data(self.data_merging_artifact.merged_data_file_path)

            # Split the data into training and validation sets
            logging.info("Splitting data into training and testing sets")
            df, df_test = split_data(df, train_size=self.data_splitting_config.train_test_split_ratio)

            # Split the data into training and testing sets
            logging.info("Splitting data into training and testing sets")
            df_train, df_validation = split_data(df, test_size=self.data_splitting_config.validation_split_ratio)

            # Create DataSplittingArtifact
            data_splitting_artifact = DataSplittingArtifact(
                train_data_file_path=self.data_splitting_config.train_data_file_path,
                validation_data_file_path=self.data_splitting_config.validation_data_file_path,
                test_data_file_path=self.data_splitting_config.test_data_file_path,
            )

            # Data shape information
            logging.info(f"Shape of training data: {df_train.shape}")
            logging.info(f"Shape of validation data: {df_validation.shape}")
            logging.info(f"Shape of testing data: {df_test.shape}")

            # Save all dataframes
            logging.info("Saving training, testing, and validation data")
            save_data(df_train, self.data_splitting_config.train_data_file_path)
            save_data(df_validation, self.data_splitting_config.validation_data_file_path)
            save_data(df_test, self.data_splitting_config.test_data_file_path)

            return data_splitting_artifact
        except Exception as e:
            raise APException(e, sys)
import sys

from ap_gpt.entity.artifact_entity import DataMergingArtifact, DataSplittingArtifact
from ap_gpt.entity.config_entity import DataSplittingConfig
from ap_gpt.utils.main_utils import read_data, split_data, save_data
from ap_gpt.ap_logger import logging
from ap_gpt.ap_exception import APException


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
            df_val, df = split_data(df, self.data_splitting_config.validation_split_ratio)

            # Split the data into training and testing sets
            logging.info("Splitting data into training and testing sets")
            df_train, df_test = split_data(df, self.data_splitting_config.train_test_split_ratio)


            # Create DataSplittingArtifact
            data_splitting_artifact = DataSplittingArtifact(
                train_data_file_path=self.data_splitting_config.train_data_file_path,
                test_data_file_path=self.data_splitting_config.test_data_file_path,
                validation_data_file_path=self.data_splitting_config.validation_data_file_path,
            )

            # Save all dataframes
            logging.info("Saving training, testing, and validation data")
            save_data(df_train, data_splitting_artifact.train_data_file_path)
            save_data(df_test, data_splitting_artifact.test_data_file_path)
            save_data(df_val, data_splitting_artifact.validation_data_file_path)

            return data_splitting_artifact
        except Exception as e:
            raise APException(e, sys)
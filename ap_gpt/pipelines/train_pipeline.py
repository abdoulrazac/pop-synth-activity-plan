import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from ap_gpt.ap_logger import logging
from ap_gpt.ap_exception import APException
from ap_gpt.components.data_ingestion import DataIngestion
from ap_gpt.components.data_merging import DataMerging
from ap_gpt.components.data_splitting import DataSplitting
from ap_gpt.components.data_to_sequence import DataToSequence
from ap_gpt.components.data_tokenizer import DataTokenizer
from ap_gpt.components.household_data_processing import HouseholdDataProcessing
from ap_gpt.components.person_data_processing import PersonDataProcessing
from ap_gpt.components.trip_data_processing import TripDataProcessing
from ap_gpt.entity.artifact_entity import (
    DataIngestionArtifact, DataProcessingArtifact, DataMergingArtifact,
    DataSplittingArtifact, DataTokenizerArtifact, DataToSequenceArtifact
)
from ap_gpt.entity.config_entity import (
    DataIngestionConfig, DataProcessingConfig, DataMergingConfig,
    DataSplittingConfig, DataTokenizerConfig, DataToSequenceConfig
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
        self.data_processing_config: DataProcessingConfig = DataProcessingConfig()
        self.data_merging_config: DataMergingConfig = DataMergingConfig()
        self.data_splitting_config: DataSplittingConfig = DataSplittingConfig()
        self.data_tokenizer_config: DataTokenizerConfig = DataTokenizerConfig()
        self.data_to_sequence_config: DataToSequenceConfig = DataToSequenceConfig()
        self.data_transformation_config = None
        self.model_trainer_config = None
        self.model_evaluation_config = None
        self.model_pusher_config = None

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_data_preprocessing(self, data_ingestion_artifact: DataIngestionArtifact) -> DataProcessingArtifact:
        """
        This method of TrainPipeline class is responsible for starting data preprocessing
        """
        try:
            logging.info("Entered the start_data_preprocessing method of TrainPipeline class")
            # Implement data preprocessing logic here
            household_data_processing = HouseholdDataProcessing(
                data_processing_config=self.data_processing_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            person_data_processing = PersonDataProcessing(
                data_processing_config=self.data_processing_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            trip_data_processing = TripDataProcessing(
                data_processing_config=self.data_processing_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            logging.info("Preprocessing data using multithreading")
            with ThreadPoolExecutor() as executor:
                household_future = executor.submit(household_data_processing.initiate_data_processing)
                person_future = executor.submit(person_data_processing.initiate_data_processing)
                trip_future = executor.submit(trip_data_processing.initiate_data_processing)

                household_data_processing_artifact = household_future.result()
                person_data_processing_artifact = person_future.result()
                trip_data_processing_artifact = trip_future.result()

            data_processing_artifact = DataProcessingArtifact(
                household_processed_data_file_path=household_data_processing_artifact.household_processed_data_file_path,
                person_processed_data_file_path=person_data_processing_artifact.person_processed_data_file_path,
                trip_processed_data_file_path=trip_data_processing_artifact.trip_processed_data_file_path,
            )

            logging.info(
                "Exited the start_data_preprocessing method of TrainPipeline class"
            )
            return data_processing_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_data_merging(self, data_processing_artifact: DataProcessingArtifact) -> DataMergingArtifact:
        """
        This method of TrainPipeline class is responsible for starting data merging
        """
        try:
            logging.info("Entered the start_data_merging method of TrainPipeline class")
            # Implement data merging logic here
            data_merging = DataMerging(data_processing_artifact=data_processing_artifact,
                                       data_merging_config=self.data_merging_config)
            data_merging_artifact = data_merging.initiate_data_merging()
            logging.info("Exited the start_data_merging method of TrainPipeline class")

            return data_merging_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_data_splitting(self, data_merging_artifact: DataMergingArtifact) -> DataSplittingArtifact:
        """
        This method of TrainPipeline class is responsible for starting data splitting
        """
        try:
            logging.info("Entered the start_data_splitting method of TrainPipeline class")
            # Implement data splitting logic here
            data_splitting = DataSplitting(data_merging_artifact=data_merging_artifact,
                                           data_splitting_config=self.data_splitting_config)
            data_splitting_artifact = data_splitting.initiate_data_splitting()
            logging.info("Exited the start_data_splitting method of TrainPipeline class")

            return data_splitting_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_data_tokenization(self, data_splitting_artifact: DataSplittingArtifact) -> DataTokenizerArtifact :
        """
        This method of TrainPipeline class is responsible for starting data tokenization
        """
        try:
            logging.info("Entered the start_data_tokenization method of TrainPipeline class")
            # Implement data tokenization logic here

            data_tokenizer = DataTokenizer(
                data_splitting_artifact=data_splitting_artifact,
                data_tokenizer_config=self.data_tokenizer_config
            )
            data_tokenizer_artifact = data_tokenizer.initiate_tokenization()
            logging.info("Exited the start_data_tokenization method of TrainPipeline class")

            return data_tokenizer_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_data_to_sequence(self, data_tokenizer_artifact : DataTokenizerArtifact) -> DataToSequenceArtifact :
        """
        This method of TrainPipeline class is responsible for starting data to sequence conversion
        """
        try:
            logging.info("Entered the start_data_to_sequence method of TrainPipeline class")
            # Implement data to sequence conversion logic here
            data_to_sequence = DataToSequence(
                data_tokenizer_artifact=data_tokenizer_artifact,
                data_to_sequence_config=self.data_to_sequence_config
            )
            data_to_sequence_artifact = data_to_sequence.initiate_data_to_sequence()
            logging.info("Exited the start_data_to_sequence method of TrainPipeline class")
            return data_to_sequence_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_model_trainer(self) -> None:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        pass

    def start_model_evaluation(self) -> None:
        """
        This method of TrainPipeline class is responsible for starting model evaluation
        """
        pass

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running the entire pipeline
        """
        try:
            logging.info("==================================================================")
            logging.info("      Entered the run_pipeline method of TrainPipeline class     ")
            logging.info("==================================================================")

            logging.info("===> Executing data ingestion <===")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion completed successfully")

            logging.info("===> Executing data preprocessing <===")
            data_processing_artifact = self.start_data_preprocessing(data_ingestion_artifact)
            logging.info("Data preprocessing completed successfully")

            logging.info("===> Executing data merging <===")
            data_merging_artifact = self.start_data_merging(data_processing_artifact)
            logging.info("Data merging completed successfully")

            logging.info("===> Executing data splitting <===")
            data_splitting_artifact = self.start_data_splitting(data_merging_artifact)
            logging.info("Data splitting completed successfully")

            logging.info("===> Executing data tokenization <===")
            data_tokenizer_artifact = self.start_data_tokenization(data_splitting_artifact)
            logging.info("Data tokenization completed successfully")

            # ----> A supprimer après test <---- #

            # data_tokenizer_artifact = DataTokenizerArtifact(
            #     tokenizer_file_path=self.data_tokenizer_config.tokenizer_file_path,
            #     train_encoded_data_file_path=self.data_tokenizer_config.train_encoded_data_file_path,
            #     test_encoded_data_file_path=self.data_tokenizer_config.test_encoded_data_file_path,
            #     pad_token_idx=self.data_tokenizer_config.pad_token_idx,
            #     nb_actions=45
            # )

            #------------------------------------#


            logging.info("===> Executing data to sequence conversion <===")
            data_to_sequence_artifact = self.start_data_to_sequence(data_tokenizer_artifact=data_tokenizer_artifact)
            logging.info("Data to sequence conversion completed successfully")

            logging.info("==================================================================")
            logging.info("      Exited the run_pipeline method of TrainPipeline class       ")
            logging.info("==================================================================")
        except Exception as e:
            raise APException(e, sys)

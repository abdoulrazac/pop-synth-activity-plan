import sys

from ap_gpt.ap_logger import logging
from ap_gpt.ap_exception import APException
from ap_gpt.components.data_ingestion import DataIngestion
from ap_gpt.entity.artifact_entity import DataIngestionArtifact


class TrainPipeline :
    def __init__(self):
        self.data_ingestion_config = None
        self.data_validation_config = None
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

    def start_data_preprocessing(self) -> None:
        """
        This method of TrainPipeline class is responsible for starting data preprocessing
        """


    def start_data_splitting(self) -> None:
        """
        This method of TrainPipeline class is responsible for starting data splitting
        """
        pass

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
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            data_ingestion_artifact = self.start_data_ingestion()
            # Add other pipeline steps here
            logging.info(
                "Exited the run_pipeline method of TrainPipeline class"
            )
        except Exception as e:
            raise APException(e, sys)

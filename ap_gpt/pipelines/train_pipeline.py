import sys
import os

from concurrent.futures import ThreadPoolExecutor
from from_root import from_root

from ap_gpt.ap_exception import APException
from ap_gpt.ap_logger import logging
from ap_gpt.components.data_ingestion import DataIngestion
from ap_gpt.components.data_merging import DataMerging
from ap_gpt.components.data_splitting import DataSplitting
from ap_gpt.components.data_to_sequence import DataToSequence
from ap_gpt.components.data_tokenizer import DataTokenizer
from ap_gpt.components.household_data_processing import HouseholdDataProcessing
from ap_gpt.components.model_trainer import ModelTrainer
from ap_gpt.components.person_data_processing import PersonDataProcessing
from ap_gpt.components.trip_data_processing import TripDataProcessing
from ap_gpt.constants import SEARCH_GRID_FILE_PATH
from ap_gpt.entity.artifact_entity import (
    DataIngestionArtifact, DataProcessingArtifact, DataMergingArtifact,
    DataSplittingArtifact, DataTokenizerArtifact, DataToSequenceArtifact, ModelTrainerArtifact
)
from ap_gpt.entity.config_entity import (
    DataIngestionConfig, DataProcessingConfig, DataMergingConfig,
    DataSplittingConfig, DataTokenizerConfig, DataToSequenceConfig, ModelTrainerConfig, TrainingPipelineConfig
)
from ap_gpt.models.gpt_activity_plan.action_gpt import ActionGPT
from ap_gpt.utils.main_utils import read_yaml_file


class TrainPipeline:
    def __init__(self, model_name: str = "default"):
        """
        This method of TrainPipeline class is responsible for initializing the pipeline
        Args :
            model_name (str) : name of the model to be trained
        """
        self.model_name = model_name

        # Initialize the configuration classes
        self.training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()
        self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
        self.data_processing_config: DataProcessingConfig = DataProcessingConfig()
        self.data_merging_config: DataMergingConfig = DataMergingConfig()
        self.data_splitting_config: DataSplittingConfig = DataSplittingConfig()
        self.data_tokenizer_config: DataTokenizerConfig = DataTokenizerConfig()
        self.data_to_sequence_config: DataToSequenceConfig = DataToSequenceConfig()

        self._search_grid_config = read_yaml_file(os.path.join(from_root(), SEARCH_GRID_FILE_PATH))

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

    def start_save_metrics(self, model_trainer_artifact : ModelTrainerArtifact) -> None:
        """
        This method of TrainPipeline class is responsible for saving metrics
        """
        try:
            logging.info("Entered the start_save_metrics method of TrainPipeline class")

            # Check if the directory exists, if not create it
            os.makedirs(self.training_pipeline_config.metric_store_path, exist_ok=True)

            # File name
            file_name = os.path.join(self.training_pipeline_config.metric_store_path, self.model_name + ".csv")

            # Check if the file exists, if not create it
            if not os.path.isfile(file_name):
                with open(file_name, 'w') as f:
                    f.write(
                        "model_name;config;test_loss;" +
                        "action_accuracy;action_precision;action_recall;action_f1_score;" +
                        "duration_accuracy;duration_precision;duration_recall;duration_f1_score;" +
                        "distance_accuracy;distance_precision;distance_recall;distance_f1_score\n"
                    )

            # Write the metrics to the file
            with open(file_name, 'a') as f:
                f.write(
                    f"{model_trainer_artifact.model_name};{model_trainer_artifact.model_trainer_config};" +
                    f"{model_trainer_artifact.metric_artifact.best_model_test_loss};" +

                    f"{model_trainer_artifact.metric_artifact.action_metrics.accuracy};" +
                    f"{model_trainer_artifact.metric_artifact.action_metrics.precision};" +
                    f"{model_trainer_artifact.metric_artifact.action_metrics.recall};" +
                    f"{model_trainer_artifact.metric_artifact.action_metrics.f1_score};" +

                    f"{model_trainer_artifact.metric_artifact.duration_metrics.accuracy};" +
                    f"{model_trainer_artifact.metric_artifact.duration_metrics.precision};" +
                    f"{model_trainer_artifact.metric_artifact.duration_metrics.recall};" +
                    f"{model_trainer_artifact.metric_artifact.duration_metrics.f1_score};" +

                    f"{model_trainer_artifact.metric_artifact.distance_metrics.accuracy};" +
                    f"{model_trainer_artifact.metric_artifact.distance_metrics.precision};" +
                    f"{model_trainer_artifact.metric_artifact.distance_metrics.recall};" +
                    f"{model_trainer_artifact.metric_artifact.distance_metrics.f1_score}\n"
                )

            logging.info("Exited the start_save_metrics method of TrainPipeline class")
        except Exception as e:
            raise APException(e, sys)

    def start_grid_search_training(self,
                            data_tokenizer_artifact : DataTokenizerArtifact,
                            data_to_sequence_artifact : DataToSequenceArtifact,) -> None:
        """
        This method of TrainPipeline class is responsible for starting grid search
        """

        logging.info("Entered the start_grid_search_training method of TrainPipeline class")

        logging.debug(self._search_grid_config)

        list_num_layers = self._search_grid_config["list_num_layers"]
        list_embed_size = self._search_grid_config["list_embed_size"]
        list_forward_expansion = self._search_grid_config["list_forward_expansion"]
        list_dropout = self._search_grid_config["list_dropout"]
        epochs = self._search_grid_config["nb_epochs"]
        heads = self._search_grid_config["heads"]

        with ThreadPoolExecutor() as executor:
            futures = []
            for num_layers in list_num_layers:
                for embed_size in list_embed_size:
                    for forward_expansion in list_forward_expansion:
                        for dropout in list_dropout:
                            model_trainer_config = ModelTrainerConfig(
                                heads=heads,
                                model_name=f"{self.model_name}_{num_layers}_{embed_size}_{forward_expansion}_{dropout}",
                                pad_token_idx = data_tokenizer_artifact.pad_token_idx,
                                nb_actions = data_tokenizer_artifact.nb_actions,
                                name_vocab_size = data_tokenizer_artifact.name_vocab_size,
                                max_sequence_length = data_to_sequence_artifact.max_sequence_length,
                                num_layers=num_layers,
                                embed_size=embed_size,
                                forward_expansion=forward_expansion,
                                dropout=dropout,
                                epochs= epochs,
                            )
                            model_trainer = ModelTrainer(
                                model=ActionGPT(config=model_trainer_config),
                                model_trainer_config=model_trainer_config,
                                data_tokenizer_artifact=data_tokenizer_artifact,
                                data_to_sequence_artifact=data_to_sequence_artifact,
                            )
                            futures.append(executor.submit(model_trainer.initiate_training))

            for future in futures:
                model_trainer_artifact = future.result()
                self.start_save_metrics(model_trainer_artifact)

        # for num_layers in list_num_layers:
        #     for embed_size in list_embed_size:
        #         for forward_expansion in list_forward_expansion:
        #             for dropout in list_dropout:
        #                 model_trainer_config =ModelTrainerConfig(
        #                     heads=2,
        #                     model_name=f"{self.model_name}_{num_layers}_{embed_size}_{forward_expansion}_{dropout}",
        #                     pad_token_idx = data_tokenizer_artifact.pad_token_idx,
        #                     nb_actions = data_tokenizer_artifact.nb_actions,
        #                     name_vocab_size = data_tokenizer_artifact.name_vocab_size,
        #                     max_sequence_length = data_to_sequence_artifact.max_sequence_length,
        #                     num_layers=num_layers,
        #                     embed_size=embed_size,
        #                     forward_expansion=forward_expansion,
        #                     dropout=dropout,
        #                 )
        #                 model_trainer = ModelTrainer(
        #                     model=ActionGPT(config=model_trainer_config),
        #                     model_trainer_config=model_trainer_config,
        #                     data_tokenizer_artifact=data_tokenizer_artifact,
        #                     data_to_sequence_artifact=data_to_sequence_artifact,
        #                 )
        #                 model_trainer_artifact = model_trainer.initiate_training()
        #
        #                 # Save the model_trainer_artifact
        #                 self.start_save_metrics(model_trainer_artifact)

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running the entire pipeline
        """
        try:
            logging.info("==================================================================")
            logging.info("      Entered the run_pipeline method of TrainPipeline class     ")
            logging.info("==================================================================")

            # logging.info("===> Executing data ingestion <===")
            # data_ingestion_artifact = self.start_data_ingestion()
            # logging.info("Data ingestion completed successfully")
            #
            # logging.info("===> Executing data preprocessing <===")
            # data_processing_artifact = self.start_data_preprocessing(data_ingestion_artifact)
            # logging.info("Data preprocessing completed successfully")
            #
            # logging.info("===> Executing data merging <===")
            # data_merging_artifact = self.start_data_merging(data_processing_artifact)
            # logging.info("Data merging completed successfully")
            #
            # logging.info("===> Executing data splitting <===")
            # data_splitting_artifact = self.start_data_splitting(data_merging_artifact)
            # logging.info("Data splitting completed successfully")
            #
            # logging.info("===> Executing data tokenization <===")
            # data_tokenizer_artifact = self.start_data_tokenization(data_splitting_artifact)
            # logging.info("Data tokenization completed successfully")
            #
            # logging.info("===> Executing data to sequence conversion <===")
            # data_to_sequence_artifact = self.start_data_to_sequence(data_tokenizer_artifact=data_tokenizer_artifact)
            # logging.info("Data to sequence conversion completed successfully")

            # ----> A supprimer après test <---- #

            data_tokenizer_artifact = DataTokenizerArtifact(
                tokenizer_file_path=self.data_tokenizer_config.tokenizer_file_path,
                train_encoded_data_file_path=self.data_tokenizer_config.train_encoded_data_file_path,
                test_encoded_data_file_path=self.data_tokenizer_config.test_encoded_data_file_path,
                pad_token_idx=(81, 97, 139),
                nb_actions=45,
                name_vocab_size = {'action': 13, 'duration': 45, 'distance': 49}
            )

            data_to_sequence_artifact = DataToSequenceArtifact(
                train_x_data_as_sequence_file_path=self.data_to_sequence_config.train_x_data_as_sequence_file_path,
                train_y_data_as_sequence_file_path=self.data_to_sequence_config.train_y_data_as_sequence_file_path,
                test_x_data_as_sequence_file_path=self.data_to_sequence_config.test_x_data_as_sequence_file_path,
                test_y_data_as_sequence_file_path=self.data_to_sequence_config.test_y_data_as_sequence_file_path,
                max_sequence_length=150,
            )

            #------------------------------------#

            logging.info("===> Executing search grid training <===")
            self.start_grid_search_training(
                data_tokenizer_artifact=data_tokenizer_artifact,
                data_to_sequence_artifact=data_to_sequence_artifact,
            )
            logging.info("Search grid training completed successfully")

            logging.info("==================================================================")
            logging.info("      Exited the run_pipeline method of TrainPipeline class       ")
            logging.info("==================================================================")
        except Exception as e:
            raise APException(e, sys)

import os
import sys

import numpy as np
from from_root import from_root

from ap.ap_exception import APException
from ap.ap_logger import logging
from ap.components.data_ingestion import DataIngestion
from ap.components.data_merging import DataMerging
from ap.components.data_splitting import DataSplitting
from ap.components.data_to_sequence import DataToSequence
from ap.components.data_tokenizer import DataTokenizer
from ap.components.household_data_processing import HouseholdDataProcessing
from ap.components.model_selection import ModelSelection
from ap.components.model_trainer import ModelTrainer
from ap.components.person_data_processing import PersonDataProcessing
from ap.components.trip_data_processing import TripDataProcessing
from ap.constants import SEARCH_GRID_FILE_PATH, ModelName
from ap.entity.artifact_entity import (
    DataIngestionArtifact, DataProcessingArtifact, DataMergingArtifact,
    DataSplittingArtifact, DataTokenizerArtifact, DataToSequenceArtifact, ModelTrainerArtifact, ModelSelectionArtifact
)
from ap.entity.config_entity import (
    DataIngestionConfig, DataProcessingConfig, DataMergingConfig,
    DataSplittingConfig, DataTokenizerConfig, DataToSequenceConfig, ModelTrainerConfig, TrainingPipelineConfig,
    ModelSelectionConfig
)
from ap.models.gpt_activity_plan.action_gpt import ActionGPT
from ap.models.lstm_activity_plan.action_lstm import ActionLSTM
from ap.utils.main_utils import read_yaml_file


class TrainPipeline:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        """
        This method of TrainPipeline class is responsible for initializing the pipeline
        Args :
            model_name (str) : name of the model to be trained
        """

        # Initialize the configuration classes
        self.training_pipeline_config = training_pipeline_config
        self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        self.data_processing_config: DataProcessingConfig = DataProcessingConfig(training_pipeline_config=training_pipeline_config)
        self.data_merging_config: DataMergingConfig = DataMergingConfig(training_pipeline_config=training_pipeline_config)
        self.data_splitting_config: DataSplittingConfig = DataSplittingConfig(training_pipeline_config=training_pipeline_config)
        self.data_tokenizer_config: DataTokenizerConfig = DataTokenizerConfig(training_pipeline_config=training_pipeline_config)
        self.data_to_sequence_config: DataToSequenceConfig = DataToSequenceConfig(training_pipeline_config=training_pipeline_config)
        self.model_selection_config: ModelSelectionConfig = ModelSelectionConfig(training_pipeline_config=training_pipeline_config)

        self.model_name = self.training_pipeline_config.model_name
        self.metric_file_name = os.path.join(self.training_pipeline_config.metric_store_path, self.model_name + ".csv")

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

            logging.info("Preprocessing data using")

            household_data_processing_artifact = household_data_processing.initiate_data_processing()
            person_data_processing_artifact = person_data_processing.initiate_data_processing()
            trip_data_processing_artifact = trip_data_processing.initiate_data_processing()

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

    def start_data_tokenization(self,
                                data_merging_artifact: DataMergingArtifact,
                                data_splitting_artifact: DataSplittingArtifact,
                                ) -> DataTokenizerArtifact:
        """
        This method of TrainPipeline class is responsible for starting data tokenization
        """
        try:
            logging.info("Entered the start_data_tokenization method of TrainPipeline class")
            # Implement data tokenization logic here

            data_tokenizer = DataTokenizer(
                data_merging_artifact=data_merging_artifact,
                data_splitting_artifact=data_splitting_artifact,
                data_tokenizer_config=self.data_tokenizer_config
            )
            data_tokenizer_artifact = data_tokenizer.initiate_tokenization()
            logging.info("Exited the start_data_tokenization method of TrainPipeline class")

            return data_tokenizer_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_data_to_sequence(self,
                               data_merging_artifact: DataMergingArtifact,
                               data_tokenizer_artifact: DataTokenizerArtifact
                               ) -> DataToSequenceArtifact:
        """
        This method of TrainPipeline class is responsible for starting data to sequence conversion
        """
        try:
            logging.info("Entered the start_data_to_sequence method of TrainPipeline class")
            # Implement data to sequence conversion logic here
            data_to_sequence = DataToSequence(
                data_merging_artifact=data_merging_artifact,
                data_tokenizer_artifact=data_tokenizer_artifact,
                training_pipeline_config=self.training_pipeline_config,
                data_to_sequence_config=self.data_to_sequence_config
            )
            data_to_sequence_artifact = data_to_sequence.initiate_data_to_sequence()
            logging.info("Exited the start_data_to_sequence method of TrainPipeline class")
            return data_to_sequence_artifact
        except Exception as e:
            raise APException(e, sys)

    def start_save_metrics(self, model_trainer_artifact: ModelTrainerArtifact) -> None:
        """
        This method of TrainPipeline class is responsible for saving metrics
        """
        try:
            logging.info("Entered the start_save_metrics method of TrainPipeline class")

            # Check if the directory exists, if not create it
            os.makedirs(self.training_pipeline_config.metric_store_path, exist_ok=True)

            # File name

            # Compute mean
            f1_score_mean = np.mean([
                model_trainer_artifact.metric_artifact.action_metrics.f1_score,
                model_trainer_artifact.metric_artifact.duration_metrics.f1_score,
                model_trainer_artifact.metric_artifact.distance_metrics.f1_score
            ])

            # Check if the file exists, if not create it
            if not os.path.isfile(self.metric_file_name):
                with open(self.metric_file_name, 'w') as f:
                    f.write(
                        "model_name;model_path;config;test_loss;" +
                        "action_accuracy;action_precision;action_recall;action_f1_score;" +
                        "duration_accuracy;duration_precision;duration_recall;duration_f1_score;" +
                        "distance_accuracy;distance_precision;distance_recall;distance_f1_score;f1_score_mean\n"
                    )

            # Write the metrics to the file
            with open(self.metric_file_name, 'a') as f:
                f.write(
                    f"{self.model_name};" +
                    f"{model_trainer_artifact.trained_model_file_path};{model_trainer_artifact.model_trainer_config};" +
                    f"{model_trainer_artifact.metric_artifact.best_model_validation_loss};" +

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
                    f"{model_trainer_artifact.metric_artifact.distance_metrics.f1_score};" +
                    f"{f1_score_mean}\n"
                )

            logging.info("Exited the start_save_metrics method of TrainPipeline class")
        except Exception as e:
            raise APException(e, sys)

    def run_grid_search_gpt(self,
                            data_merging_artifact: DataMergingArtifact,
                            data_tokenizer_artifact: DataTokenizerArtifact,
                            data_to_sequence_artifact: DataToSequenceArtifact,
                            ) -> None:
        """
        This method of TrainPipeline class is responsible for starting grid search for GPT model
        """

        logging.info("Entered the start_grid_search_training method of TrainPipeline class for GPT model")

        list_num_layers = self._search_grid_config["gpt"]["list_num_layers"]
        list_embed_size = self._search_grid_config["gpt"]["list_embed_size"]
        list_hidden_dim = self._search_grid_config["gpt"]["list_hidden_dim"]
        list_forward_expansion = self._search_grid_config["gpt"]["list_forward_expansion"]
        list_dropout = self._search_grid_config["gpt"]["list_dropout"]
        epochs = self._search_grid_config["gpt"]["nb_epochs"]
        heads = self._search_grid_config["gpt"]["heads"]
        batch_size = self._search_grid_config["gpt"]["batch_size"]


        # -------- A SUPPRIMER APRÈS -------- #
        # Get Env variable NUM_LAYERS_INDEX
        num_layers_index = int(os.getenv("NUM_LAYERS_INDEX")) if os.getenv("NUM_LAYERS_INDEX") else 0
        list_num_layers = list_num_layers[num_layers_index:num_layers_index+1]
        # ------------------------------------ #

        for num_layers in list_num_layers:
            for embed_size in list_embed_size:
                for forward_expansion in list_forward_expansion:
                    for hidden_dim in list_hidden_dim:
                        for dropout in list_dropout:
                            model_trainer_config = ModelTrainerConfig(
                                heads=heads,
                                model_name=f"{self.model_name}_{num_layers}_{embed_size}_{forward_expansion}_{hidden_dim}_{dropout}",
                                pad_token_idx=data_tokenizer_artifact.pad_token_idx,
                                nb_actions=data_tokenizer_artifact.nb_actions,
                                vocab_size=data_tokenizer_artifact.vocab_size,
                                name_vocab_size=data_tokenizer_artifact.name_vocab_size,
                                max_sequence_length=data_to_sequence_artifact.max_sequence_length,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim,
                                embed_size=embed_size,
                                forward_expansion=forward_expansion,
                                dropout=dropout,
                                epochs=epochs,
                                batch_size=batch_size,
                                training_pipeline_config=self.training_pipeline_config
                            )
                            model_trainer = ModelTrainer(
                                model=ActionGPT(model_trainer_config=model_trainer_config).to(model_trainer_config.device),
                                model_trainer_config=model_trainer_config,
                                data_tokenizer_artifact=data_tokenizer_artifact,
                                data_to_sequence_artifact=data_to_sequence_artifact,
                                data_merging_artifact=data_merging_artifact,
                            )

                            model_trainer_artifact = model_trainer.initiate_training()
                            self.start_save_metrics(model_trainer_artifact)

    def run_grid_search_lstm(self,
                            data_merging_artifact: DataMergingArtifact,
                            data_tokenizer_artifact: DataTokenizerArtifact,
                            data_to_sequence_artifact: DataToSequenceArtifact,
                            ) -> None:
        """
        This method of TrainPipeline class is responsible for starting grid search for LSTM model
        """

        logging.info("Entered the start_grid_search_training method of TrainPipeline class for LSTM model")

        list_num_layers = self._search_grid_config["lstm"]["list_num_layers"]
        list_embed_size = self._search_grid_config["lstm"]["list_embed_size"]
        list_dropout = self._search_grid_config["lstm"]["list_dropout"]
        epochs = self._search_grid_config["lstm"]["nb_epochs"]
        list_hidden_dim = self._search_grid_config["lstm"]["list_hidden_dim"]
        batch_size = self._search_grid_config["lstm"]["batch_size"]

        # -------- A SUPPRIMER APRÈS -------- #
        # Get Env variable NUM_LAYERS_INDEX
        num_layers_index = int(os.getenv("NUM_LAYERS_INDEX")) if os.getenv("NUM_LAYERS_INDEX") else 0
        list_num_layers = list_num_layers[num_layers_index:num_layers_index+1]
        # ------------------------------------ #

        for num_layers in list_num_layers:
            for embed_size in list_embed_size:
                for hidden_dim in list_hidden_dim:
                    for dropout in list_dropout:

                        model_trainer_config = ModelTrainerConfig(
                            hidden_dim=hidden_dim,
                            model_name=f"{self.model_name}_{num_layers}_{embed_size}_{hidden_dim}_{dropout}",
                            pad_token_idx=data_tokenizer_artifact.pad_token_idx,
                            nb_actions=data_tokenizer_artifact.nb_actions,
                            vocab_size=data_tokenizer_artifact.vocab_size,
                            name_vocab_size=data_tokenizer_artifact.name_vocab_size,
                            max_sequence_length=data_to_sequence_artifact.max_sequence_length,
                            num_layers=num_layers,
                            embed_size=embed_size,
                            dropout=dropout,
                            epochs=epochs,
                            batch_size=batch_size,
                            training_pipeline_config=self.training_pipeline_config
                        )
                        model_trainer = ModelTrainer(
                            model=ActionLSTM(model_trainer_config=model_trainer_config).to(model_trainer_config.device),
                            model_trainer_config=model_trainer_config,
                            data_tokenizer_artifact=data_tokenizer_artifact,
                            data_to_sequence_artifact=data_to_sequence_artifact,
                            data_merging_artifact=data_merging_artifact,
                        )

                        model_trainer_artifact = model_trainer.initiate_training()
                        self.start_save_metrics(model_trainer_artifact)

    def start_grid_search_training(self,
                                   data_merging_artifact: DataMergingArtifact,
                                   data_tokenizer_artifact: DataTokenizerArtifact,
                                   data_to_sequence_artifact: DataToSequenceArtifact,
                                   ) -> None:
        """
        This method of TrainPipeline class is responsible for starting grid search
        """

        logging.info("Entered the start_grid_search_training method of TrainPipeline class")

        if self.model_name == ModelName.GPT.value:
            self.run_grid_search_gpt(data_merging_artifact, data_tokenizer_artifact, data_to_sequence_artifact)
        elif self.model_name == ModelName.LSTM.value:
            self.run_grid_search_lstm(data_merging_artifact, data_tokenizer_artifact, data_to_sequence_artifact)
        else:
            raise APException("Invalid model name", sys)


    def start_model_selection(self,
                              data_merging_artifact: DataMergingArtifact,
                              data_tokenizer_artifact: DataTokenizerArtifact,
                              data_to_sequence_artifact: DataToSequenceArtifact,
                              ) -> ModelSelectionArtifact:
        """
        This method of TrainPipeline class is responsible for starting model selection
        """
        try:
            logging.info("Entered the start_model_selection method of TrainPipeline class")
            # Implement model selection logic here

            model_selection = ModelSelection(
                metric_file_path=self.metric_file_name,
                data_merging_artifact=data_merging_artifact,
                data_tokenizer_artifact=data_tokenizer_artifact,
                data_to_sequence_artifact=data_to_sequence_artifact,
                model_selection_config=self.model_selection_config,
            )

            model_selection_artifact = model_selection.initiate_model_selection()
            logging.info("Model selection completed successfully")

            logging.info("Exited the start_model_selection method of TrainPipeline class")
            return model_selection_artifact
        except Exception as e:
            raise APException(e, sys)

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running the entire pipeline
        """
        try:
            logging.info("==================================================================")
            logging.info("      Entered the run_pipeline method of TrainPipeline class     ")
            logging.info("==================================================================")

            # ----> A supprimer après test <---- #

            #
            # base_path = '/Users/abdoul/Desktop/these/Activity-Plan/artifact/ActionGPT/'
            # # base_path = '/Users/doctorant/Desktop/These/pop-synth-activity-plan/artifact/ActionGPT/'
            #
            # data_merging_artifact = DataMergingArtifact(
            #     merged_data_file_path= base_path + '/data/merged_data.parquet',
            #     household_columns_number=6,
            #     person_columns_number=12,
            #     trip_columns_number=135,
            #     household_columns=['household_size', 'number_of_bikes', 'number_of_vehicles',
            #                        'house_occupation_type', 'has_internet', 'house_type'],
            #     person_columns=['link_ref_person', 'socioprofessional_class', 'is_adolescent',
            #                     'employed', 'sex', 'age', 'school_level', 'studies', 'has_pt_subscription',
            #                     'travel_respondent', 'has_license', 'number_of_trips'],
            #     trip_columns=[f'{label}_{i}' for i in range(45) for label in ['action', 'duration', 'distance']]
            # )
            #
            # data_tokenizer_artifact = DataTokenizerArtifact(
            #     tokenizer_file_path= base_path + '/data/tokenizer.txt',
            #     train_encoded_data_file_path= base_path + '/data/train_encoded_data.npy',
            #     validation_encoded_data_file_path= base_path + '/data/validation_encoded_data.npy',
            #     test_encoded_data_file_path= base_path + '/data/test_encoded_data.npy',
            #     pad_token_idx=(80, 95, 138),
            #     nb_actions=45,
            #     vocab_size=171,
            #     name_vocab_size={'action': 13, 'duration': 45, 'distance': 49}
            # )
            #
            # data_to_sequence_artifact = DataToSequenceArtifact(
            #     train_x_data_as_sequence_file_path= base_path + '/data/X_train_data_as_sequence.npy',
            #     train_y_data_as_sequence_file_path= base_path + '/data/Y_train_data_as_sequence.npy',
            #     validation_x_data_as_sequence_file_path= base_path + '/data/X_validation_data_as_sequence.npy',
            #     validation_y_data_as_sequence_file_path= base_path + '/data/Y_validation_data_as_sequence.npy',
            #     test_x_data_as_sequence_file_path= base_path + '/data/X_test_data_as_sequence.npy',
            #     max_sequence_length=152
            # )

            # ------------------------------------#

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
            data_tokenizer_artifact = self.start_data_tokenization(
                data_merging_artifact=data_merging_artifact,
                data_splitting_artifact=data_splitting_artifact
            )
            logging.info("Data tokenization completed successfully")

            logging.info("===> Executing data to sequence conversion <===")
            data_to_sequence_artifact = self.start_data_to_sequence(
                data_merging_artifact=data_merging_artifact,
                data_tokenizer_artifact=data_tokenizer_artifact
            )
            logging.info("Data to sequence conversion completed successfully")

            logging.info("===> Executing search grid training <===")
            self.start_grid_search_training(
                data_merging_artifact=data_merging_artifact,
                data_tokenizer_artifact=data_tokenizer_artifact,
                data_to_sequence_artifact=data_to_sequence_artifact,
            )
            logging.info("Search grid training completed successfully")

            logging.info("===> Executing model selection and data generating <===")
            model_selection_artifact = self.start_model_selection(
                data_merging_artifact=data_merging_artifact,
                data_tokenizer_artifact=data_tokenizer_artifact,
                data_to_sequence_artifact=data_to_sequence_artifact,
            )
            logging.info("Model selection and data generating completed successfully")

            logging.info("==================================================================")
            logging.info("      Exited the run_pipeline method of TrainPipeline class       ")
            logging.info("==================================================================")
        except Exception as e:
            raise APException(e, sys)

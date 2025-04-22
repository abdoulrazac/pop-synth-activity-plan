import os
import sys
from typing import Tuple, Literal

import pandas as pd
from from_root import from_root

from ap_gpt.ap_exception import APException
from ap_gpt.ap_logger import logging

from ap_gpt.components.model_trainer import ModelTrainer
from ap_gpt.constants import GENERATING_PARAM_GRID_FILE_PATH
from ap_gpt.entity.artifact_entity import DataTokenizerArtifact, DataToSequenceArtifact, ModelSelectionArtifact, \
    DataMergingArtifact
from ap_gpt.entity.config_entity import ModelSelectionConfig, ModelTrainerConfig, TrainingPipelineConfig
from ap_gpt.models.gpt_activity_plan.action_gpt import ActionGPT
from ap_gpt.utils.main_utils import read_data, read_yaml_file, save_data, write_yaml_file


class ModelSelection :

    model_trainer : ModelTrainer
    model_config : ModelTrainerConfig
    best_model_path : str = ""
    best_model_config : str = ""
    best_model_metric : str = ""

    def __init__(self,
                 metric_file_path : str,
                 data_merging_artifact : DataMergingArtifact,
                 data_tokenizer_artifact : DataTokenizerArtifact,
                 data_to_sequence_artifact: DataToSequenceArtifact,
                 model_selection_config : ModelSelectionConfig = ModelSelectionConfig(),
                 ) -> None:
        self.metric_file_path = metric_file_path
        self.data_tokenizer_artifact = data_tokenizer_artifact
        self.data_merging_artifact = data_merging_artifact
        self.data_to_sequence_artifact = data_to_sequence_artifact
        self.model_selection_config = model_selection_config

        self._generating_param_grid = read_yaml_file(os.path.join(from_root(), GENERATING_PARAM_GRID_FILE_PATH))

    def get_best_model_based_on_metric(self,
                                       metric : Tuple[Literal["min", "max"], str] = ("max", "f1_score_mean")
                                       ) -> None:
        """
        This function will get the best model from the metric file
        """
        try :
            # Read the metric file
            df = read_data(self.metric_file_path, sep=";")

            # Check if the metric is present in the file
            if metric[1] not in df.columns:
                raise ValueError(f"{metric[1]} is not present in the file")

            # Get the best model based on the metric
            if metric[0] == "max":
                best_model = df.loc[df[metric[1]].idxmax()]
            elif metric[0] == "min":
                best_model = df.loc[df[metric[1]].idxmin()]
            else:
                raise ValueError(f"{metric[0]} is not a valid metric type")

            # Get the best model path
            self.best_model_config = best_model["config"]
            self.best_model_path = best_model["model_path"]


            config = eval(self.best_model_config)
            self.model_config = ModelTrainerConfig( # noqa
                model_name=config["model_name"],
                heads=int(config["heads"]),
                pad_token_idx=(int(config["pad_token_idx"][0]), int(config["pad_token_idx"][1]), int(config["pad_token_idx"][2])),
                nb_actions=int(config["nb_actions"]),
                name_vocab_size=dict(config["name_vocab_size"]),
                max_sequence_length=int(config["max_sequence_length"]),
                embed_size=int(config["embed_size"]),
                num_layers=int(config["num_layers"]),
                forward_expansion=int(config["forward_expansion"]),
                dropout=float(config["dropout"]),
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                verbose=bool(config["verbose"]),
            )

            self.model_trainer = ModelTrainer(
                model=ActionGPT(config=self.model_config),
                model_trainer_config=self.model_config,
                data_tokenizer_artifact=self.data_tokenizer_artifact,
                data_to_sequence_artifact=self.data_to_sequence_artifact,
                data_merging_artifact=self.data_merging_artifact,
            )
            self.model_trainer.load_model(self.best_model_path)
            self.model_trainer.model.to(self.model_config.device)
        except Exception as e:
            raise APException(e, sys) from e

    def generate_data(self, input_data : pd.DataFrame) -> None:

        try:
            # Get the generating param grid
            list_temperature = self._generating_param_grid["list_temperature"]
            list_do_sample = self._generating_param_grid["list_do_sample"]
            list_top_k = self._generating_param_grid["list_top_k"]

            # Return the model
            for temperature in list_temperature:
                for do_sample in list_do_sample:
                    for top_k in list_top_k:
                        df = self.model_trainer.generate(
                            X=input_data,
                            temperature=temperature,
                            do_sample=do_sample,
                            top_k=top_k
                        )

                        # Save the file
                        save_data(df, os.path.join(
                            self.model_selection_config.data_generated_store_path,
                            f"{temperature}_{do_sample}_{top_k}.parquet"
                        ))
        except Exception as e:
            raise APException(e, sys) from e

    def initiate_model_selection(self) -> ModelSelectionArtifact:

        # Read the input data
        input_data = read_data(self.data_to_sequence_artifact.test_x_data_as_sequence_file_path, is_array=True)

        # Get the best model based on the metric
        self.get_best_model_based_on_metric()

        # Generate the data
        self.generate_data(input_data)

        # Create the model selection artifact
        model_selection_artifact = ModelSelectionArtifact(
            model_path=self.best_model_path,
            model_config=self.best_model_config,
            data_generated_store_path=self.model_selection_config.data_generated_store_path
        )

        # Save the model selection artifact
        write_yaml_file(
            file_path=self.model_selection_config.data_generated_detail_file_path,
            content={
                "model_path": self.best_model_path,
                "model_config": self.best_model_config,
                "data_generated_store_path": self.model_selection_config.data_generated_store_path
            }
        )

        return model_selection_artifact
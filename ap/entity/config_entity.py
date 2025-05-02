import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

from from_root import from_root

from ap.constants import *
from ap.utils.main_utils import get_device

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:

    def __init__(self, model_name: ModelName = ModelName.GPT) -> None:
        self.model_name: str = model_name.value
        self.timestamp: str = TIMESTAMP
        self.artifact_dir_name: str = os.path.join(from_root(), ARTIFACT_DIR_NAME, self.model_name, self.timestamp)
        self.data_store_path: str = os.path.join(self.artifact_dir_name, DATA_STORE_DIR_NAME)
        self.metric_store_path: str = os.path.join(self.artifact_dir_name, METRIC_STORE_DIR_NAME)

@dataclass
class DataIngestionConfig :

    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.training_pipeline_config = training_pipeline_config

        # Raws DATA
        self.data_store_path: str = training_pipeline_config.data_store_path

        self.household_raw_data_file_path: str = os.path.join(
            DATA_RAW_DIR_NAME, DATA_RAW_HOUSEHOLD_FILE_NAME
        )
        self.person_raw_data_file_path: str = os.path.join(
            DATA_RAW_DIR_NAME, DATA_RAW_PERSON_FILE_NAME
        )
        self.trip_raw_data_file_path: str = os.path.join(
            DATA_RAW_DIR_NAME, DATA_RAW_TRIP_FILE_NAME
        )

        # Store DATA
        self.household_data_file_path: str = os.path.join(
            training_pipeline_config.artifact_dir_name,
            DATA_STORE_DIR_NAME, DATA_HOUSEHOLD_FILE_NAME
        )

        self.person_data_file_path: str = os.path.join(
            training_pipeline_config.artifact_dir_name,
            DATA_STORE_DIR_NAME, DATA_PERSON_FILE_NAME
        )

        self.trip_data_file_path: str = os.path.join(
            training_pipeline_config.artifact_dir_name,
            DATA_STORE_DIR_NAME, DATA_TRIP_FILE_NAME
        )


@dataclass
class DataValidationConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.data_store_path: str = training_pipeline_config.data_store_path
        self.validation_report_file_path: str = os.path.join(
            self.data_store_path, DATA_VALIDATION_REPORT_FILE_NAME
        )


@dataclass
class DataProcessingConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.data_store_path: str = training_pipeline_config.data_store_path
        self.household_processed_data_file_path: str = os.path.join(
            self.data_store_path, DATA_HOUSEHOLD_PROCESSED_FILE_NAME
        )
        self.person_processed_data_file_path: str =  os.path.join(
            self.data_store_path, DATA_PERSON_PROCESSED_FILE_NAME
        )
        self.trip_processed_data_file_path: str = os.path.join(
            self.data_store_path, DATA_TRIP_PROCESSED_FILE_NAME
        )

@dataclass
class DataMergingConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.data_store_path: str = training_pipeline_config.data_store_path
        self.merged_data_file_path: str = os.path.join(
            self.data_store_path, DATA_MERGED_FILE_NAME
        )

@dataclass
class DataSplittingConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.data_store_path: str = training_pipeline_config.data_store_path
        self.train_data_file_path: str = os.path.join(
            self.data_store_path, TRAIN_DATA_FILE_NAME
        )
        self.test_data_file_path: str = os.path.join(
            self.data_store_path, TEST_DATA_FILE_NAME
        )
        self.validation_data_file_path: str = os.path.join(
            self.data_store_path, VALIDATION_DATA_FILE_NAME
        )
        self.train_test_split_ratio: float = TRAIN_TEST_SPLIT_RATIO
        self.validation_split_ratio: float = VALIDATION_SPLIT_RATIO

@dataclass
class DataToSequenceConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.data_store_path: str = training_pipeline_config.data_store_path
        self.train_x_data_as_sequence_file_path: str = os.path.join(
            self.data_store_path, "X_" + TRAIN_DATA_AS_SEQUENCE_FILE_NAME
        )
        self.train_y_data_as_sequence_file_path: str = os.path.join(
            self.data_store_path, "Y_" + TRAIN_DATA_AS_SEQUENCE_FILE_NAME
        )

        self.test_x_data_as_sequence_file_path: str = os.path.join(
            self.data_store_path, "X_" + TEST_DATA_AS_SEQUENCE_FILE_NAME
        )

        self.test_y_data_as_sequence_file_path: str = os.path.join(
            self.data_store_path, "Y_" + TEST_DATA_AS_SEQUENCE_FILE_NAME
        )
        self.validation_x_data_as_sequence_file_path: str = os.path.join(
            self.data_store_path, "X_" + VALIDATION_DATA_AS_SEQUENCE_FILE_NAME
        )
        self.validation_y_data_as_sequence_file_path: str = os.path.join(
            self.data_store_path, "Y_" + VALIDATION_DATA_AS_SEQUENCE_FILE_NAME
        )
        self.action_nb_cols : int = ACTION_NB_COLS
        self.drop_pad: bool = True

@dataclass
class DataTokenizerConfig:

    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.data_store_path: str = training_pipeline_config.data_store_path
        self.tokenizer_file_path: str = os.path.join(
            self.data_store_path, TOKENIZER_FILE_NAME
        )
        self.train_encoded_data_file_path : str = os.path.join(
            self.data_store_path, TRAIN_ENCODED_DATA_FILE_NAME
        )
        self.validation_encoded_data_file_path : str = os.path.join(
            self.data_store_path, VALIDATION_ENCODED_DATA_FILE_NAME
        )
        self.test_encoded_data_file_path : str = os.path.join(
            self.data_store_path, TEST_ENCODED_DATA_FILE_NAME
        )

@dataclass
class ModelTrainerConfig :
    best_model_path : str
    pad_token_idx : Tuple[int, int, int]
    heads : int
    max_sequence_length : int
    name_vocab_size : Dict[str, int]
    nb_actions : int
    vocab_size : int
    embed_size : int = 2
    dropout : float =0.1
    forward_expansion : int = 4
    num_layers : int = 1
    hidden_dim : int = 128
    epochs : int = 1 # 00
    batch_size : int = 128
    verbose : bool = False
    device : str = get_device()

    def __init__(self,
                 model_name:str,
                 pad_token_idx: Tuple[int, int, int],
                 nb_actions: int,
                 vocab_size: int,
                 name_vocab_size: Dict[str, int],
                 max_sequence_length: int,
                 heads: int = 1,
                 embed_size: int = 2,
                 num_layers: int = 1,
                 hidden_dim: int = 128,
                 forward_expansion: int = 4,
                 dropout: float = 0.1,
                 epochs : int = 1,
                 batch_size: int = 128,
                 verbose: bool = False,
                 training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()
            ) -> None:
        self.heads = heads
        self.pad_token_idx = pad_token_idx
        self.nb_actions = nb_actions
        self.vocab_size = vocab_size
        self.name_vocab_size = name_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_name = training_pipeline_config.model_name

        self.model_store_path: str = os.path.join(
            training_pipeline_config.artifact_dir_name,
            MODEL_STORE_DIR_NAME
        )

        self.best_model_path = os.path.join(
            self.model_store_path, model_name + "_best_model.pth"
        )
        self.benchmark_file_path = os.path.join(
            self.model_store_path, model_name + "_benchmark.json"
        )

    def __repr__(self) :
        return (f"ModelConfig(\nembed_size={self.embed_size}, \nheads={self.heads}, \ndropout={self.dropout}, " +
                f"\nforward_expansion={self.forward_expansion}, \nmax_len={self.max_sequence_length}, " +
                f"\nnum_layers={self.num_layers}, \nvocab_size={self.vocab_size}, \ndevice={self.device})" +
                f"\npad_token_idx={self.pad_token_idx}, \nepochs={self.epochs}, \nhidden_dim={self.hidden_dim}, " +
                f"\nbatch_size={self.batch_size}, \nmodel_store_path={self.model_store_path}, " +
                f"\nmodel_name={self.model_name}, \nnb_actions={self.nb_actions}, " +
                f"\nname_vocab_size={self.name_vocab_size}, \nverbose={self.verbose})")

    def to_json(self) -> str:

        data = {
            "pad_token_idx": self.pad_token_idx,
            "heads": self.heads,
            "embed_size": self.embed_size,
            "dropout": self.dropout,
            "forward_expansion": self.forward_expansion,
            "max_len": self.max_sequence_length,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "name_vocab_size": self.name_vocab_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "model_store_path": self.model_store_path,
            "model_name": self.model_name,
            "nb_actions": self.nb_actions,
            "verbose": self.verbose,
            "device": str(self.device),
            "max_sequence_length": self.max_sequence_length,
        }

        return json.dumps(data)


@dataclass
class ModelSelectionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()) -> None:
        self.data_generated_store_path: str = os.path.join(
            training_pipeline_config.artifact_dir_name,
            DATA_GENERATED_STORE_DIR_NAME
        )
        self.data_generated_detail_file_path: str = os.path.join(
            self.data_generated_store_path,
            DATA_GENERATED_DETAIL_FILE_NAME
        )
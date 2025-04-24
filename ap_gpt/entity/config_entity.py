from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple
from from_root import from_root

from ap_gpt.constants import *
from ap_gpt.utils.main_utils import get_device

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    model_name: str
    artifact_dir_name: str = os.path.join(from_root(), ARTIFACT_DIR_NAME) #, TIMESTAMP)
    timestamp: str = TIMESTAMP
    metric_store_path: str = os.path.join(artifact_dir_name, METRIC_STORE_DIR_NAME)

    def __init__(self, model_name: str = "ActionGPT") -> None:
        self.model_name = model_name
        self.artifact_dir_name = os.path.join(from_root(), ARTIFACT_DIR_NAME, model_name)
        self.metric_store_path = os.path.join(self.artifact_dir_name, METRIC_STORE_DIR_NAME)
        self.timestamp = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig :
    # Raws DATA
    data_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME
    )
    household_raw_data_file_path: str = os.path.join(
        DATA_RAW_DIR_NAME, DATA_RAW_HOUSEHOLD_FILE_NAME
    )
    person_raw_data_file_path: str = os.path.join(
        DATA_RAW_DIR_NAME, DATA_RAW_PERSON_FILE_NAME
    )
    trip_raw_data_file_path: str = os.path.join(
        DATA_RAW_DIR_NAME, DATA_RAW_TRIP_FILE_NAME
    )

    # Store DATA
    household_data_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME, DATA_HOUSEHOLD_FILE_NAME
    )

    person_data_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME, DATA_PERSON_FILE_NAME
    )

    trip_data_file_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME, DATA_TRIP_FILE_NAME
    )


@dataclass
class DataValidationConfig:
    data_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME
    )
    validation_report_file_path: str = os.path.join(
        data_store_path, DATA_VALIDATION_REPORT_FILE_NAME
    )


@dataclass
class DataProcessingConfig:
    data_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME
    )
    household_processed_data_file_path: str = os.path.join(
        data_store_path, DATA_HOUSEHOLD_PROCESSED_FILE_NAME
    )
    person_processed_data_file_path: str =  os.path.join(
        data_store_path, DATA_PERSON_PROCESSED_FILE_NAME
    )
    trip_processed_data_file_path: str = os.path.join(
        data_store_path, DATA_TRIP_PROCESSED_FILE_NAME
    )

@dataclass
class DataMergingConfig:
    data_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME
    )
    merged_data_file_path: str = os.path.join(
        data_store_path, DATA_MERGED_FILE_NAME
    )

@dataclass
class DataSplittingConfig:
    data_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME
    )
    train_data_file_path: str = os.path.join(
        data_store_path, TRAIN_DATA_FILE_NAME
    )
    test_data_file_path: str = os.path.join(
        data_store_path, TEST_DATA_FILE_NAME
    )
    validation_data_file_path: str = os.path.join(
        data_store_path, VALIDATION_DATA_FILE_NAME
    )
    train_test_split_ratio: float = TRAIN_TEST_SPLIT_RATIO
    validation_split_ratio: float = VALIDATION_SPLIT_RATIO

@dataclass
class DataToSequenceConfig:
    data_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME
    )
    train_x_data_as_sequence_file_path: str = os.path.join(
        data_store_path, "X_" + TRAIN_DATA_AS_SEQUENCE_FILE_NAME
    )
    train_y_data_as_sequence_file_path: str = os.path.join(
        data_store_path, "Y_" + TRAIN_DATA_AS_SEQUENCE_FILE_NAME
    )

    test_x_data_as_sequence_file_path: str = os.path.join(
        data_store_path, "X_" + TEST_DATA_AS_SEQUENCE_FILE_NAME
    )

    test_y_data_as_sequence_file_path: str = os.path.join(
        data_store_path, "Y_" + TEST_DATA_AS_SEQUENCE_FILE_NAME
    )
    validation_x_data_as_sequence_file_path: str = os.path.join(
        data_store_path, "X_" + VALIDATION_DATA_AS_SEQUENCE_FILE_NAME
    )
    validation_y_data_as_sequence_file_path: str = os.path.join(
        data_store_path, "Y_" + VALIDATION_DATA_AS_SEQUENCE_FILE_NAME
    )
    action_nb_cols : int = ACTION_NB_COLS
    drop_pad: bool = True

@dataclass
class DataTokenizerConfig:
    data_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        DATA_STORE_DIR_NAME
    )
    tokenizer_file_path: str = os.path.join(
        data_store_path, TOKENIZER_FILE_NAME
    )
    train_encoded_data_file_path : str = os.path.join(
        data_store_path, TRAIN_ENCODED_DATA_FILE_NAME
    )
    validation_encoded_data_file_path : str = os.path.join(
        data_store_path, VALIDATION_ENCODED_DATA_FILE_NAME
    )
    test_encoded_data_file_path : str = os.path.join(
        data_store_path, TEST_ENCODED_DATA_FILE_NAME
    )

@dataclass
class ModelTrainerConfig :
    best_model_path : str
    pad_token_idx : Tuple[int, int, int]
    heads : int
    max_sequence_length : int
    name_vocab_size : Dict[str, int]
    nb_actions : int
    embed_size : int = 2
    dropout : float =0.2
    forward_expansion : int = 4
    num_layers : int = 1
    vocab_size : int = 256
    epochs : int = 1 # 00
    batch_size : int = 128
    verbose : bool = False
    device : str = get_device()
    model_store_path: str = os.path.join(
        training_pipeline_config.artifact_dir_name,
        MODEL_STORE_DIR_NAME
    )

    def __init__(self,
                 model_name:str,
                 heads: int,
                 pad_token_idx: Tuple[int, int, int],
                 nb_actions: int,
                 name_vocab_size: Dict[str, int],
                 max_sequence_length: int,
                 embed_size: int = 2,
                 num_layers: int = 1,
                 forward_expansion: int = 4,
                 dropout: float = 0.2,
                 epochs : int = 1,
                 batch_size: int = 128,
                 verbose: bool = False,
            ) -> None:
        self.heads = heads
        self.pad_token_idx = pad_token_idx
        self.nb_actions = nb_actions
        self.name_vocab_size = name_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_name = model_name

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
                f"\npad_token_idx={self.pad_token_idx}, \naction_start_idx={self.action_start_idx}, \nepochs={self.epochs}" +
                f"\nbatch_size={self.batch_size}, \nmodel_store_path={self.model_store_path}, " +
                f"\nmodel_name={self.model_name}, \nnb_actions={self.nb_actions}, " +
                f"\nname_vocab_size={self.name_vocab_size}, \nverbose={self.verbose})")

    def to_json(self) -> str:
        return (
            "{"
            f'"pad_token_idx": {self.pad_token_idx}, '
            f'"heads": {self.heads}, '
            f'"embed_size": {self.embed_size}, '
            f'"dropout": {self.dropout}, '
            f'"forward_expansion": {self.forward_expansion}, '
            f'"max_len": {self.max_sequence_length}, '
            f'"num_layers": {self.num_layers}, '
            f'"vocab_size": {self.vocab_size}, '
            f'"name_vocab_size": {self.name_vocab_size}, '
            f'"epochs": {self.epochs}, '
            f'"batch_size": {self.batch_size}, '
            f'"model_store_path": "{self.model_store_path}",'
            f'"model_name": "{self.model_name}", '
            f'"nb_actions": {self.nb_actions}, '
            f'"verbose": {self.verbose}, '
            f'"device": "{self.device}", '
            f'"max_sequence_length": {self.max_sequence_length}, '
            "}"
        )


@dataclass
class ModelSelectionConfig:
    data_generated_store_path: str = os.path.join(training_pipeline_config.artifact_dir_name, DATA_GENERATED_STORE_DIR_NAME)
    data_generated_detail_file_path: str = os.path.join(data_generated_store_path, DATA_GENERATED_DETAIL_FILE_NAME)
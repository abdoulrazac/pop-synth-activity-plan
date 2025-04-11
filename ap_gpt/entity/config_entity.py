from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple
from from_root import from_root

from ap_gpt.constants import *
from ap_gpt.utils.main_utils import get_device

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    artifact_dir_name: str = os.path.join(from_root(), ARTIFACT_DIR_NAME) #, TIMESTAMP)
    timestamp: str = TIMESTAMP


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

    max_seq_len : int = MAX_SEQ_LENGTH
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
    test_encoded_data_file_path : str = os.path.join(
        data_store_path, TEST_ENCODED_DATA_FILE_NAME
    )

@dataclass
class ModelConfig :
    pad_token_idx : Tuple[int, int, int]
    embed_size : int
    heads : int
    dropout : float =0.2
    best_model_path : str = "best_model.pth"
    forward_expansion : int = 4
    max_len : int = MAX_SEQ_LENGTH
    num_layers : int = 1
    vocab_size : int = 256
    name_vocab_size : Dict[str, int] = {"action" : 256, "duration" : 256, "distance" : 256},
    action_start_idx : int = 19
    epochs : int = 100
    batch_size : int = 32
    device : str = get_device()

    def __repr__(self) :
        return (f"ModelConfig(\nembed_size={self.embed_size}, \nheads={self.heads}, \ndropout={self.dropout}, " +
                f"\nforward_expansion={self.forward_expansion}, \nmax_len={self.max_len}, " +
                f"\nnum_layers={self.num_layers}, \nvocab_size={self.vocab_size}, \ndevice={self.device})" +
                f"\npad_token_idx={self.pad_token_idx}, \naction_start_idx={self.action_start_idx}, \nepochs={self.epochs}")

@dataclass
class ModelTrainerConfig :
    best_model_path: str
    benchmark_file_path: str


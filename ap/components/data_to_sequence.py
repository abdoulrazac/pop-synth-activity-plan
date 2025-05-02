import sys
from typing import Union, List, Tuple

import pandas as pd
from from_root import from_root

from ap.ap_exception import APException
from ap.ap_logger import logging
from ap.components.data_tokenizer import DataTokenizer
from ap.constants import *
from ap.entity.artifact_entity import DataTokenizerArtifact, DataToSequenceArtifact, \
    DataMergingArtifact
from ap.entity.config_entity import DataToSequenceConfig, TrainingPipelineConfig
from ap.utils.main_utils import read_yaml_file, read_data, save_data, pad_sequence


class DataToSequence:
    def __init__(self,
                 data_merging_artifact: DataMergingArtifact,
                 data_tokenizer_artifact : DataTokenizerArtifact,
                 training_pipeline_config : TrainingPipelineConfig,
                 data_to_sequence_config : DataToSequenceConfig = DataToSequenceConfig(),
                 ) -> None:
        """
        Args:
            data_tokenizer_artifact (DataTokenizerArtifact):
                DataTokenizerArtifact object containing the tokenizer file path and pad token index.
            data_to_sequence_config (DataToSequenceConfig):
                DataToSequenceConfig object containing the configuration for data to sequence conversion.
        """
        self.training_pipeline_config = training_pipeline_config
        self._schema_config = read_yaml_file(file_path=os.path.join(from_root(), SCHEMA_FILE_PATH))
        self.person_id_col = self._schema_config[TABLE_PERSON_NAME][SCHEMA_IDENTIFIER_NAME]
        self.table_name = TABLE_TRIP_NAME

        self.data_tokenizer_artifact = data_tokenizer_artifact
        self.pad_token_idx = data_tokenizer_artifact.pad_token_idx
        self.nb_actions = self.data_tokenizer_artifact.nb_actions

        self.data_merging_artifact = data_merging_artifact
        self.data_tokenizer = DataTokenizer(data_merging_artifact)
        self.data_tokenizer.load(data_tokenizer_artifact.tokenizer_file_path)

        self.data_to_sequence_config = data_to_sequence_config
        self.action_nb_cols = self.data_to_sequence_config.action_nb_cols

    def split_row_to_sequence(self,
                              sequence: Union[np.ndarray, List],
                              drop_pad: bool = True
                              ) -> Tuple[np.ndarray, np.ndarray]:

        """
        This method of DataToSequence class is responsible for splitting the row into a sequence
        """
        sequence_length = len(sequence)
        x_list, y_list = list(), list()
        # start_idx = len(sequence) - self.nb_actions * self.action_nb_cols
        start_idx = self.data_merging_artifact.household_columns_number + self.data_merging_artifact.person_columns_number
        end_idx = sequence_length - self.action_nb_cols
        for i in range(start_idx, end_idx, self.action_nb_cols):
            seq_x, seq_y = list(sequence[:i]), list(sequence[i:i + self.action_nb_cols])
            if drop_pad and ((seq_y[0] in self.pad_token_idx) or (seq_y[1] in self.pad_token_idx) or (seq_y[2] in self.pad_token_idx)):
                break
            if len(seq_x) <= end_idx :
                if self.training_pipeline_config.model_name == ModelName.LSTM.value:
                    seq_x = pad_sequence(seq_x, sequence_length, self.pad_token_idx, padding="pre")
                else:
                    seq_x = pad_sequence(seq_x, sequence_length, self.pad_token_idx, padding="post")

            x_list.append(np.array(seq_x))
            y_list.append(np.array(seq_y))
        return np.vstack(x_list), np.vstack(y_list)

    def split_data_to_sequence(self, data: Union[pd.DataFrame, np.array], drop_pad=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method of DataToSequence class is responsible for splitting the data into sequences
        """
        data = data.values if isinstance(data, pd.DataFrame) else data
        x_list, y_list = list(), list()
        for row in data:
            x, y = self.split_row_to_sequence(row, drop_pad=drop_pad)
            x_list.append(x)
            y_list.append(y)

        # Concatenate all sequences
        x_list = np.vstack(x_list)
        y_list = np.vstack(y_list)

        # Convert y_list to name index
        y_list[:, 0] = self.data_tokenizer.convert_index_to_name(name="action", index=y_list[:, 0])
        y_list[:, 1] = self.data_tokenizer.convert_index_to_name(name="duration", index=y_list[:, 1])
        y_list[:, 2] = self.data_tokenizer.convert_index_to_name(name="distance", index=y_list[:, 2])

        return x_list, y_list

    def split_test_data_to_sequence(self, data: Union[pd.DataFrame, np.array], max_sequence_length : int) -> np.ndarray :
        """
        This method of DataToSequence class is responsible for splitting the test data into sequences
        """
        data = data.values if isinstance(data, pd.DataFrame) else data
        start_idx = self.data_merging_artifact.household_columns_number + self.data_merging_artifact.person_columns_number
        x_list = data[ :, :start_idx]

        if self.training_pipeline_config.model_name == ModelName.LSTM.value:
            x_list = pad_sequence(x_list, max_sequence_length, self.pad_token_idx, padding="pre")
        else:
            x_list = pad_sequence(x_list, max_sequence_length, self.pad_token_idx, padding="post")

        return x_list

    def initiate_data_to_sequence(self) -> DataToSequenceArtifact:
        """
        This method of DataToSequence class is responsible for initiating the data to sequence process
        """
        try:
            # Read data
            logging.info("Importing datasets for sequencing")
            df_train = read_data(self.data_tokenizer_artifact.train_encoded_data_file_path)
            df_validation = read_data(self.data_tokenizer_artifact.validation_encoded_data_file_path)

            # Split data to sequence
            logging.info("Splitting data to sequence")
            df_train_x_seq, df_train_y_seq = self.split_data_to_sequence(df_train, drop_pad=self.data_to_sequence_config.drop_pad)
            df_validation_x_seq, df_validation_y_seq = self.split_data_to_sequence(df_validation, drop_pad=self.data_to_sequence_config.drop_pad)

            # Max sequence length
            max_sequence_length = df_train_x_seq.shape[1]

            # Split test data to sequence
            df_test = read_data(self.data_tokenizer_artifact.test_encoded_data_file_path)
            df_test_x_seq = self.split_test_data_to_sequence(df_test, max_sequence_length=max_sequence_length)

            data_to_sequence_artifact = DataToSequenceArtifact(
                train_x_data_as_sequence_file_path=self.data_to_sequence_config.train_x_data_as_sequence_file_path,
                train_y_data_as_sequence_file_path=self.data_to_sequence_config.train_y_data_as_sequence_file_path,
                validation_x_data_as_sequence_file_path=self.data_to_sequence_config.validation_x_data_as_sequence_file_path,
                validation_y_data_as_sequence_file_path=self.data_to_sequence_config.validation_y_data_as_sequence_file_path,
                test_x_data_as_sequence_file_path=self.data_to_sequence_config.test_x_data_as_sequence_file_path,
                max_sequence_length = max_sequence_length
            )

            # Save data to sequence
            logging.info("Saving data to sequence")
            save_data(df_train_x_seq, self.data_to_sequence_config.train_x_data_as_sequence_file_path)
            save_data(df_train_y_seq, self.data_to_sequence_config.train_y_data_as_sequence_file_path)
            save_data(df_validation_x_seq, self.data_to_sequence_config.validation_x_data_as_sequence_file_path)
            save_data(df_validation_y_seq, self.data_to_sequence_config.validation_y_data_as_sequence_file_path)
            save_data(df_test_x_seq, self.data_to_sequence_config.test_x_data_as_sequence_file_path)

            return data_to_sequence_artifact

        except Exception as e:
            raise APException(e, sys) from e

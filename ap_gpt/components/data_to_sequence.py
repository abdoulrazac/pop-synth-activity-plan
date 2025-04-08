import sys
import pandas as pd
from typing import Union, List, Tuple

from ap_gpt.ap_exception import APException
from ap_gpt.constants import *
from ap_gpt.entity.artifact_entity import DataSplittingArtifact, DataTokenizerArtifact, DataToSequenceArtifact
from ap_gpt.entity.config_entity import DataToSequenceConfig
from ap_gpt.utils.main_utils import read_yaml_file, read_data, save_data
from ap_gpt.ap_logger import logging


class DataToSequence:
    def __init__(self,
                 data_tokenizer_artifact : DataTokenizerArtifact,
                 pad_token_idx : Tuple[int, int, int],
                 data_to_sequence_config : DataToSequenceConfig = DataToSequenceConfig(),
                 ) -> None:
        """
        Args:
            data_splitting_artifact (DataSplittingArtifact):  Data splitting artifact containing the file paths for
                train, validation, and test data.
            data_transformation_config (DataToSequenceConfig): Configuration for data transformation.
        """

        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        self.data_tokenizer_artifact = data_tokenizer_artifact
        self.data_to_sequence_config = data_to_sequence_config
        self.table_name = TABLE_TRIP_NAME
        self.person_id_col = self._schema_config[TABLE_PERSON_NAME][SCHEMA_IDENTIFIER_NAME]
        self.pad_token_idx = pad_token_idx
        self.action_nb_cols = self.data_to_sequence_config.action_nb_cols
        self.nb_actions = None
        self.max_seq_len = self.data_to_sequence_config.max_seq_len
        
    def pivot_wider(self, data : pd.DataFrame) -> pd.DataFrame:

        # Pivot wider actions table to have one row per person and ordered actions
        self.nb_actions = data['person_action_id'].max()

        df_actions = data.pivot(index=self.person_id_col, columns='person_action_id').reset_index()
        df_actions.columns = [f"{x}_{y}" for x, y in df_actions.columns]
        df_actions = df_actions.rename(columns={f"{self.person_id_col}_": self.person_id_col})
        df_actions = df_actions.sort_values(by=[self.person_id_col])
        df_actions = df_actions[[self.person_id_col] + np.concatenate(
            ([[f"action_{i}", f"duration_{i}", f"distance_{i}"] for i in range(self.nb_actions)])
        ).tolist()]
        return df_actions

    def split_row_to_sequence(self,
                              sequence: Union[np.ndarray, List],
                              drop_pad: bool = True
                              ) -> Tuple[np.ndarray, np.ndarray]:

        """
        This method of DataToSequence class is responsible for splitting the row into a sequence
        """
        X, y = list(), list()
        start_idx = len(sequence) - self.nb_actions * self.action_nb_cols - 1
        end_idx = len(sequence)
        for i in range(start_idx, end_idx, self.action_nb_cols):
            seq_x, seq_y = list(sequence[:i]), list(sequence[i:i + self.action_nb_cols])
            if not (drop_pad and (seq_y[0] in self.pad_token_idx or seq_y[1] in self.pad_token_idx or seq_y[2] in self.pad_token_idx)):
                if len(seq_x) < self.max_seq_len:
                    seq_x += self.pad_token_idx * ((self.max_seq_len - len(seq_x)) // len(self.pad_token_idx))
                X.append(np.array(seq_x))
                y.append(np.array(seq_y))
            else:
                break
        return np.vstack(X), np.vstack(y)

    def split_data_to_sequence(self, data: Union[pd.DataFrame, np.array], drop_pad=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method of DataToSequence class is responsible for splitting the data into sequences
        """
        data = data.values if isinstance(data, pd.DataFrame) else data
        X, y = list(), list()
        for row in data:
            x, y_ = self.split_row_to_sequence(row, drop_pad=drop_pad)
            X.append(x)
            y.append(y_)
        return np.vstack(X), np.vstack(y)


    def initiate_data_to_sequence(self) -> DataToSequenceArtifact:
        """
        This method of DataToSequence class is responsible for initiating the data to sequence process
        """
        try:
            # Read data
            logging.info("Importing datasets for sequencing")
            df_train = read_data(self.data_tokenizer_artifact.train_encoded_data_file_path)
            df_test = read_data(self.data_tokenizer_artifact.test_encoded_data_file_path)

            # Pivot wider
            logging.info("Pivoting wider the actions table")
            df_train = self.pivot_wider(df_train)
            df_test = self.pivot_wider(df_test)

            # Split data to sequence
            logging.info("Splitting data to sequence")
            df_train_x_seq, df_train_y_seq = self.split_data_to_sequence(df_train, drop_pad=self.data_to_sequence_config.drop_pad)
            df_test_x_seq, df_test_y_seq = self.split_data_to_sequence(df_test, drop_pad=self.data_to_sequence_config.drop_pad)

            data_to_sequence_artifact = DataToSequenceArtifact(
                train_x_data_as_sequence_file_path=self.data_to_sequence_config.train_x_data_as_sequence_file_path,
                train_y_data_as_sequence_file_path=self.data_to_sequence_config.train_y_data_as_sequence_file_path,
                test_x_data_as_sequence_file_path=self.data_to_sequence_config.test_x_data_as_sequence_file_path,
                test_y_data_as_sequence_file_path=self.data_to_sequence_config.test_y_data_as_sequence_file_path
            )

            # Save data to sequence
            logging.info("Saving data to sequence")
            save_data(df_train_x_seq, data_to_sequence_artifact.train_x_data_as_sequence_file_path)
            save_data(df_train_y_seq, data_to_sequence_artifact.train_y_data_as_sequence_file_path)
            save_data(df_test_x_seq, data_to_sequence_artifact.test_x_data_as_sequence_file_path)
            save_data(df_test_y_seq, data_to_sequence_artifact.test_y_data_as_sequence_file_path)

            return data_to_sequence_artifact

        except Exception as e:
            raise APException(e, sys) from e

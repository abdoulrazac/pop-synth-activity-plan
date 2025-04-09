import sys
from typing import Optional

import pandas as pd

from ap_gpt.ap_exception import APException
from ap_gpt.ap_logger import logging
from ap_gpt.components.data_processing_base import DataProcessingBase
from ap_gpt.constants import *
from ap_gpt.entity.artifact_entity import DataIngestionArtifact, TripDataProcessingArtifact
from ap_gpt.entity.config_entity import DataProcessingConfig
from ap_gpt.utils.main_utils import read_data, save_data


class TripDataProcessing(DataProcessingBase):
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_processing_config: DataProcessingConfig = DataProcessingConfig(),
                 ) -> None:
        """
        :param data_ingestion_artifact:
        :param data_processing_config:
        """
        try:
            super().__init__()
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_processing_config = data_processing_config
            self.table_name = TABLE_TRIP_NAME
        except Exception as e:
            raise APException(e, sys)

    @staticmethod
    def cut_duration(duration: pd.Series) -> pd.Series:
        bins = np.concatenate((
            np.arange(0, 1e-3, 1e-3),
            np.arange(1e-3, 26, 5),
            np.arange(30, 180, 15),
            np.arange(180, 480, 30),
            np.arange(480, 1440, 60),
            np.arange(1440, 2880, 120),
            np.arange(2880, 4320, 180),
        )).tolist()
        return pd.cut(
            (duration / 60),
            bins=bins,
            include_lowest=True,
            labels=[f"{x // 60:02.0f}h{x % 60:02.0f}" for x in bins[1:]]
        ).astype(str)

    @staticmethod
    def cut_distance(distance: pd.Series) -> pd.Series:
        bins = np.concatenate((
            np.arange(0, 1e-3, 1e-3),
            np.arange(1e-3, 5, 0.5),
            np.arange(5, 10, 1),
            np.arange(10, 20, 2),
            np.arange(20, 50, 5),
            np.arange(50, 100, 10),
            np.arange(100, 200, 20),
            np.arange(200, 500, 50),
            np.arange(500, 1000, 100),
            np.arange(1000, 2000, 200),
        )).tolist()

        return pd.cut(
            (distance / 1_000),
            bins=bins,
            include_lowest=True,
            labels=[f"{x:06.1f}" for x in bins[1:]]
        ).astype(str)

    def recode_trip_into_action(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method Name :   recode_trip_into_action
        Description :   This method recodes the trip into actions

        Args
            data (pd.DataFrame): Dataframe to recode

        Returns:
            pd.DataFrame: Dataframe with recoded trip into actions
        """

        try:
            person_id_col = self._schema_config[TABLE_PERSON_NAME][SCHEMA_IDENTIFIER_NAME]

            data = data.sort_values(by=[person_id_col, 'departure_time'])
            data['trip_rank'] = data.sort_values(by=[person_id_col, 'departure_time']).groupby(
                person_id_col).cumcount()

            trip_actions = dict()

            ## Initial purpose
            trip_actions["initial_purpose"] = data.loc[
                data['trip_rank'] == 0, [person_id_col, 'trip_rank', 'preceding_purpose', 'departure_time']]
            trip_actions["initial_purpose"] = trip_actions["initial_purpose"].rename(
                columns={'preceding_purpose': 'action', 'departure_time': 'duration'})
            trip_actions["initial_purpose"]['distance'] = 0
            trip_actions["initial_purpose"] = trip_actions["initial_purpose"][
                [person_id_col, 'trip_rank', 'action', 'duration', 'distance']]
            trip_actions["initial_purpose"]['trip_action_rank'] = 0

            ## Current travel mode
            trip_actions["mode"] = data[[person_id_col, 'trip_rank', 'mode', 'trip_duration', 'euclidean_distance']]
            trip_actions["mode"] = trip_actions["mode"].rename(
                columns={'mode': 'action', 'euclidean_distance': 'distance', 'trip_duration': 'duration'})
            trip_actions["mode"]['trip_action_rank'] = 1

            ## Current activity
            trip_actions["activity"] = data[
                [person_id_col, 'trip_rank', 'following_purpose', 'activity_duration', 'arrival_time', 'is_last_trip']]
            trip_actions["activity"] = trip_actions["activity"].rename(
                columns={'following_purpose': 'action', 'activity_duration': 'duration'})
            trip_actions["activity"]['distance'] = 0
            trip_actions["activity"]['duration'] = trip_actions["activity"][
                ['duration', 'arrival_time', 'is_last_trip']].apply(
                lambda x: 24 * 60 * 60 - x.iloc[1] if x.iloc[2] == 1 else x.iloc[0],
                axis=1)
            trip_actions["activity"]['duration'] = trip_actions["activity"]['duration'].apply(
                lambda x: (4 * 60 * 60 + x) if x < 0 else x)  # A revoir avec les en
            trip_actions["activity"] = trip_actions["activity"][
                [person_id_col, 'trip_rank', 'action', 'duration', 'distance']]
            trip_actions["activity"]['trip_action_rank'] = 2

            ## Merge all actions
            df_actions = pd.concat(trip_actions.values())
            df_actions = df_actions.sort_values(by=[person_id_col, 'trip_rank', 'trip_action_rank'])
            df_actions['person_action_id'] = df_actions.sort_values(
                by=[person_id_col, 'trip_rank', 'trip_action_rank']).groupby(person_id_col).cumcount()
            df_actions = df_actions[[person_id_col, 'person_action_id', 'action', 'duration', 'distance']]

            return df_actions

        except Exception as e:
            raise APException(e, sys)

    def add_end_of_action(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method Name : add_end_of_action
        Description : This method adds an end-of-trip (EOT) token to the dataframe for each person.

        Args:
            data (pd.DataFrame): Dataframe containing trip actions.

        Returns:
            pd.DataFrame: Dataframe with added end-of-trip tokens.
        """

        def add_eot(dat: pd.DataFrame) -> pd.DataFrame:
            """
            Helper function to add end-of-trip token to a single person's data.

            Args:
                dat (pd.DataFrame): Dataframe for a single person.

            Returns:
                pd.DataFrame: Dataframe with added end-of-trip token.
            """
            max_rank = dat['person_action_id'].max()
            eot = dat.loc[dat['person_action_id'] == max_rank].copy()
            eot.iloc[:, 2:] = EOT_TOKEN.values()
            eot['person_action_id'] = max_rank + 1
            return pd.concat([dat, eot], ignore_index=True)

        try:
            person_id_col = self._schema_config[TABLE_PERSON_NAME][SCHEMA_IDENTIFIER_NAME]
            return data.groupby(by=[person_id_col], as_index=False).apply(add_eot).reset_index(drop=True)
        except Exception as e:
            raise APException(e, sys)

    def initiate_data_processing(self) -> Optional[TripDataProcessingArtifact]:
        """
        Method Name :   initiate_data_processing
        Description :   This method initiates the data processing component for the pipeline
        """

        try :
            logging.info("Entered trip initiate_data_processing method of DataProcessing class")

            # Read the data from the CSV files
            logging.info("Reading data from CSV files")
            df_trip = read_data(self.data_ingestion_artifact.trip_data_file_path)

            # Keep only the required columns
            logging.info("Keeping only the trip's required columns")
            df_trip = self.keep_required_columns(df_trip, TABLE_TRIP_NAME)

            # Validate the data
            logging.info("Validating trip data")
            is_valid_data = self.is_valid_data(df_trip, self.table_name, other_columns=TABLE_TRIP_REQUIRED_COLUMNS)
            if not is_valid_data:
                raise ValueError(f"Invalid data in trip data: {self.table_name}")

            # recode categorical trip columns
            logging.info("Recode trip columns (action + duration + distance + add_end_action")
            df_trip = self.recode_trip_into_action(df_trip)
            df_trip[TABLE_TRIP_DURATION_NAME] = self.cut_duration(df_trip[TABLE_TRIP_DURATION_NAME])
            df_trip[TABLE_TRIP_DISTANCE_NAME] = self.cut_distance(df_trip[TABLE_TRIP_DISTANCE_NAME])
            df_trip = self.add_end_of_action(df_trip)

            logging.info("Create trip DataProcessing artifact")
            data_processing_artifact = TripDataProcessingArtifact(
                trip_processed_data_file_path = self.data_processing_config.trip_processed_data_file_path,
            )

            # Save trip data
            logging.info("Saving trip data")
            save_data(df_trip, data_processing_artifact.trip_processed_data_file_path)

            return data_processing_artifact

        except Exception as e :
            APException(e, sys)

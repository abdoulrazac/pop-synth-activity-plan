import os
import numpy as np
# Set seed for reproducibility
np.random.seed(42)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

try:
    ITER_VALUE : int = 100 * (1 + int(os.environ['SLURM_NODEID'])) + int(os.environ['SLURM_PROCID']) + 1
except KeyError :
    ITER_VALUE : int = np.random.randint(2000, 10000)

# DATA ENGINEERING  ---------------------------------------------------------
MAX_TRIP_NUMBER = 12
RANDOM_SEED = 123

# TOKENS -------------------------------------------------------------------
PAD_TOKEN = {"ACTION" : "<ACTION_PAD>", "DURATION" : "<DURATION_PAD>", "DISTANCE" : "<DISTANCE_PAD>"}
UNK_TOKEN = "<UNK>"
SOT_TOKEN = "<SOT>"
EOT_TOKEN = {"ACTION" : "<ACTION_EOT>", "DURATION" : "<DURATION_EOT>", "DISTANCE" : "<DISTANCE_EOT>"}

# Pipeline constants ------------------------------------------------------
ARTIFACT_DIR_NAME = "artifact"

# SCHEMA FILES ------------------------------------------------------
SCHEMA_FILE_PATH = os.path.join("config", "schema.yml")
SEARCH_GRID_FILE_PATH = os.path.join("config", "search_grid.yml")
GENERATING_PARAM_GRID_FILE_PATH = os.path.join("config", "generating_param_grid.yml")

# Table constants ------------------------------------------------------
TABLE_HOUSEHOLD_NAME = "households"
TABLE_PERSON_NAME = "persons"
TABLE_TRIP_NAME = "trips"

SCHEMA_IDENTIFIER_NAME = "identifier"
SCHEMA_WEIGHT_NAME = "weight"
SCHEMA_NUMERICAL_NAME = "numerical"
SCHEMA_CATEGORICAL_NAME = "categorical"
SCHEMA_RELATIONAL_NAME = "relational"
SCHEMA_CUTTING_NAME = "cutting"
SCHEMA_RECODING_NAME = "recoding"
SCHEMA_REMOVE_NAME = "to_remove"

TABLE_TRIP_DURATION_NAME = "duration"
TABLE_TRIP_DISTANCE_NAME = "distance"
TABLE_TRIP_REQUIRED_COLUMNS = ["preceding_purpose", "departure_time", 'mode', 'trip_duration', 'euclidean_distance', 'following_purpose', 'activity_duration', 'arrival_time', 'is_last_trip']

TABLE_PERSON_NUMBER_OF_TRIPS_NAME = "number_of_trips"

# DATA INGESTION ------------------------------------------------------
DATA_STORE_DIR_NAME = "data"
MODEL_STORE_DIR_NAME = "model"
METRIC_STORE_DIR_NAME = "metrics"
DATA_RAW_DIR_NAME = "data/raw"
DATA_GENERATED_STORE_DIR_NAME = "generated"
DATA_GENERATED_DETAIL_FILE_NAME = "generated_data_details.yml"

DATA_VALIDATION_REPORT_FILE_NAME = "validation_report.json"

DATA_RAW_HOUSEHOLD_FILE_NAME = "household.csv"
DATA_RAW_PERSON_FILE_NAME = "persons.csv"
DATA_RAW_TRIP_FILE_NAME = "trips.csv"

DATA_HOUSEHOLD_FILE_NAME = "household.parquet"
DATA_PERSON_FILE_NAME = "person.parquet"
DATA_TRIP_FILE_NAME = "trip.parquet"

DATA_HOUSEHOLD_PROCESSED_FILE_NAME = "processed_household.parquet"
DATA_PERSON_PROCESSED_FILE_NAME = "processed_person.parquet"
DATA_TRIP_PROCESSED_FILE_NAME = "processed_trip.parquet"

DATA_MERGED_FILE_NAME = "merged_data.parquet"

TRAIN_DATA_FILE_NAME = "train_data.parquet"
TEST_DATA_FILE_NAME = "test_data.parquet"
VALIDATION_DATA_FILE_NAME = "validation_data.parquet"

TRAIN_DATA_AS_SEQUENCE_FILE_NAME = "train_data_as_sequence.npy"
TEST_DATA_AS_SEQUENCE_FILE_NAME = "test_data_as_sequence.npy"
VALIDATION_DATA_AS_SEQUENCE_FILE_NAME = "validation_data_as_sequence.npy"

TRAIN_TEST_SPLIT_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.2

# MODEL TRAINING ------------------------------------------------------

TOKENIZER_FILE_NAME = "tokenizer.txt"
TRAIN_ENCODED_DATA_FILE_NAME = "train_encoded_data.npy"
VALIDATION_ENCODED_DATA_FILE_NAME = "validation_encoded_data.npy"
TEST_ENCODED_DATA_FILE_NAME = "test_encoded_data.npy"

# ACTION -------------------------------------------------------

ACTION_NB_COLS = 3

# MODEL TRAINING ------------------------------------------------------


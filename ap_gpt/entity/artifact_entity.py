from dataclasses import dataclass
from typing import Union

@dataclass
class DataIngestionArtifact:
    household_data_file_path: str
    person_data_file_path: str
    trip_data_file_path: str

@dataclass
class DataValidationArtifact:
    is_data_valid: bool
    validation_report_file_path: str


@dataclass
class HouseholdDataProcessingArtifact:
    household_processed_data_file_path: str


@dataclass
class PersonDataProcessingArtifact:
    person_processed_data_file_path: str


@dataclass
class TripDataProcessingArtifact:
    trip_processed_data_file_path: str


@dataclass
class DataProcessingArtifact:
    household_processed_data_file_path: str
    person_processed_data_file_path: str
    trip_processed_data_file_path: str


@dataclass
class DataMergingArtifact:
    merged_data_file_path: str

@dataclass
class DataSplittingArtifact:
    train_data_file_path: str
    test_data_file_path: str
    validation_data_file_path: str

@dataclass
class DataToSequenceArtifact:
    train_x_data_as_sequence_file_path: str
    train_y_data_as_sequence_file_path : str
    test_x_data_as_sequence_file_path: str
    test_y_data_as_sequence_file_path : str



@dataclass
class DataTokenizerArtifact:
    tokenizer_file_path: str
    train_encoded_data_file_path: str
    test_encoded_data_file_path: str

@dataclass
class MetricArtifact:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: dict

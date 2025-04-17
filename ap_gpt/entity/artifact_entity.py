from dataclasses import dataclass
from typing import Dict, Tuple


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
    max_sequence_length: int



@dataclass
class DataTokenizerArtifact:
    tokenizer_file_path: str
    train_encoded_data_file_path: str
    test_encoded_data_file_path: str
    pad_token_idx: Tuple[int, int, int]
    nb_actions: int
    name_vocab_size: Dict[str, int]

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float

@dataclass
class MetricArtifact:
    action_metrics: ModelMetrics
    duration_metrics: ModelMetrics
    distance_metrics: ModelMetrics
    best_model_test_loss : float

    def to_json(self) :
        return {
            "action_metrics": {
                "accuracy": self.action_metrics.accuracy,
                "precision": self.action_metrics.precision,
                "recall": self.action_metrics.recall,
                "f1_score": self.action_metrics.f1_score
            },
            "duration_metrics": {
                "accuracy": self.duration_metrics.accuracy,
                "precision": self.duration_metrics.precision,
                "recall": self.duration_metrics.recall,
                "f1_score": self.duration_metrics.f1_score
            },
            "distance_metrics": {
                "accuracy": self.distance_metrics.accuracy,
                "precision": self.distance_metrics.precision,
                "recall": self.distance_metrics.recall,
                "f1_score": self.distance_metrics.f1_score
            },
            "best_model_test_loss": self.best_model_test_loss
        }



@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: MetricArtifact
    model_trainer_config: str
    model_name: str

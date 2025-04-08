"""
This script is a template for creating a Python module.
"""

# Import necessary libraries
import os
import logging
from pathlib import Path
from typing import List, Tuple

# Project name
project_name = "ap_gpt" # Activity Plan GPT


# List of tuples containing file paths and a boolean indicating if the path is a directory
shared_files : List[Tuple[str, bool]] = [
    # Data directories
    ('data', True),
    ('data/raw', True),

    # Notebooks directories
    ('notebooks', True),

    # Config directories
    ('config', True),
    ('config/schema.yml', False),
    ('config/model.yml', False),

    # Constants directories
    ('constants', True),
    ('constants/__init__.py', False),

    # Pipelines directories
    ('main.py', False),
    ('setup.py', False),
    ('demo.py', False),
    ('requirements.txt', False),
    ('README.md', False)
]


ap_files : List[Tuple[str, bool]] = [

    # Constants directories
    ('constants', True),
    ('constants/__init__.py', False),

    # Pipeline directories
    ('pipelines', True),
    ('pipelines/__init__.py', False),
    ('pipelines/train_pipeline.py', False),
    ('pipelines/generate_pipeline.py', False),
    ('pipelines/evaluation_pipeline.py', False),

    # Components directories
    ('components', True),
    ('components/__init__.py', False),
    ('components/data_ingestion.py', False),
    ('components/data_processing_base.py', False),
    ('components/household_data_processing.py', False),
    ('components/person_data_processing.py', False),
    ('components/trip_data_processing.py', False),
    ('components/data_merging.py', False),
    ('components/data_splitting.py', False),
    ('components/model_trainer.py', False),
    ('components/model_evaluation.py', False),
    ('components/data_to_sequence.py', False),
    ('components/data_tokenizer.py', False),

    # Model directories
    ('models', True),
    ('models/ap_model_base.py', False),

    ## Model GPT
    ('models/gpt_activity_plan', True),
    ('models/gpt_activity_plan/__init__.py', False),
    ('models/gpt_activity_plan/gpt_activity_plan.py', False),
    ('models/gpt_activity_plan/action_gpt.py', False),
    ('models/gpt_activity_plan/self_attention.py', False),
    ('models/gpt_activity_plan/transformer_block.py', False),

    # Logging directories
    ('ap_logger', True),
    ('logs', True),
    ('ap_logger/__init__.py', False),

    # Utils directories
    ('utils', True),
    ('utils/__init__.py', False),
    ('utils/main_utils.py', False),
    ('utils/data_loader.py', False),
    ('utils/progress_bar_logger.py', False),
    ('utils/recode_functions.py', False),
    ('utils/split_data.py', False),
    ('utils/tokenizer.py', False),
    ('utils/value_prefixer.py', False),

    # Exception directories
    ('ap_exception', True),
    ('ap_exception/__init__.py', False),

    # Entity directories
    ('entity', True),
    ('entity/__init__.py', False),
    ('entity/artifact_entity.py', False),
    ('entity/config_entity.py', False),
    ('entity/estimator.py', False),
]




test_files : List[Tuple[str, bool]] = [

    # Pipeline directories
    ('pipelines', True),
    ('test_pipelines.py', False),
    ('pipelines/test_data_ingestion.py', False),
    ('pipelines/test_data_processing.py', False),
    ('pipelines/test_model_training.py', False),
    ('pipelines/test_model_evaluation.py', False),

    # Components directories
    ('components', True),
    ('test_components.py', False),
    ('components/test_data_ingestion.py', False),
    ('components/test_data_processing.py', False),
    ('components/test_model_trainer.py', False),
    ('components/test_model_evaluation.py', False),

    # Model directories
    ('models', True),
    ('models/gpt_activity_plan', True),
    ('models/test_gpt_activity_plan.py', False),
    ('models/gpt_activity_plan/test_gpt_activity_plan.py', False),
    ('models/gpt_activity_plan/test_action_gpt.py', False),
    ('models/gpt_activity_plan/test_model_config.py', False),
    ('models/gpt_activity_plan/test_self_attention.py', False),
    ('models/gpt_activity_plan/test_transformer_block.py', False),

    # Logging directories
    ('test_ap_logger.py', False),

    # Utils directories
    ('utils', True),
    ('test_utils.py', False),
    ('utils/test_main_utils.py', False),
    ('utils/test_data_loader.py', False),
    ('utils/test_prefixer.py', False),
    ('utils/test_progress_bar_logger.py', False),
    ('utils/test_recode_functions.py', False),
    ('utils/test_split_data.py', False),
    ('utils/test_tokenizer.py', False),

    # Exception directories
    ('test_ap_exception.py', False),

    # Entity directories
    ('entity', True),
    ('test_entity.py', False),
    ('entity/test_artifact_entity.py', False),
    ('entity/test_config_entity.py', False),
    ('entity/test_estimator.py', False),
]


# Function to create directories and files
def create_structure(base_path: str, files: List[Tuple[str, bool]]) -> None:
    """
    Create directories and files based on the provided structure.

    Args:
        base_path (str): The base path where the structure will be created.
        files (List[Tuple[str, bool]]): A list of tuples containing file paths and a boolean indicating if the path is a directory.
    """
    for file_path, is_dir in files:
        full_path = Path(base_path, file_path)

        if is_dir:
            # Create directory if it doesn't exist
            if not full_path.exists():
                os.makedirs(full_path)
                logging.info(f"Created directory: {full_path}")
            else:
                logging.warning(f"Directory already exists: {full_path}")
        else:
            # Create file if it doesn't exist
            if not full_path.exists():
                with open(full_path, 'w') as f:
                    pass  # Create an empty file
                logging.info(f"Created file: {full_path}")
            else:
                logging.warning(f"File already exists: {full_path}")


if __name__ == '__main__':
    # Define the base path where the structure will be created
    root_dir = os.getcwd()  # Current working directory

    logging.info(f"Base path: {root_dir}")

    # Create the shared files structure
    logging.info("===>>  Creating shared files structure...  <<===")
    create_structure(root_dir, shared_files)
    logging.info("===>> Shared files structure created successfully. <<===")

    # Create the AP files structure
    logging.info("===>>  Creating AP files structure...  <<===")
    create_structure(os.path.join(root_dir, project_name), ap_files)
    logging.info("===>> AP files structure created successfully. <<===")

    # Create test files structure
    logging.info("===>>  Creating test files structure...  <<===")
    # create_structure(os.path.join(root_dir, 'tests'), test_files)
    logging.info("===>> Test files structure created successfully. <<===")

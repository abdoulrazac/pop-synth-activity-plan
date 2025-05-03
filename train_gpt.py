from ap.constants import ModelName
from ap.entity.config_entity import TrainingPipelineConfig
from ap.pipelines.train_pipeline import TrainPipeline

train_pipeline = TrainPipeline(training_pipeline_config=TrainingPipelineConfig(model_name=ModelName.GPT))
train_pipeline.run_pipeline()
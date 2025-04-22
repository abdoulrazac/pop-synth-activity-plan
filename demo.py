from ap_gpt.entity.config_entity import TrainingPipelineConfig
from ap_gpt.pipelines.train_pipeline import TrainPipeline

# Clear a

train_pipeline = TrainPipeline(
    training_pipeline_config=TrainingPipelineConfig(model_name="ActionGPT"),
)

train_pipeline.run_pipeline()
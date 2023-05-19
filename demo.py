from census.logger import logging
from census.exception import SensorException
from census.utils import get_collection_as_dataframe
import sys,os
from census.entity.config_entity import DataIngestionConfig
from census.entity.config_entity import DataValidationConfig
from census.entity.config_entity import DataTransformationConfig
from census.entity.config_entity import ModelTrainerConfig
from census.entity.config_entity import ModelPusherConfig
from census.entity import config_entity
from census.components.data_ingestion import DataIngestion
from census.components.data_validation import DataValidation
from census.components.model_evaluation import ModelEvaluation
from census.components.data_transformation import DataTransformation
from census.components.model_trainer import ModelTrainer
from census.components.model_pusher import ModelPusher
if __name__ =="__main__":
     try:
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          data_ingestion_config = DataIngestionConfig(training_pipeline_config= training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config= data_ingestion_config )
          data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

          data_validation_config = DataValidationConfig(training_pipeline_config=  training_pipeline_config)
          data_validation = DataValidation(data_validation_config=data_validation_config , data_ingestion_artifact=data_ingestion_artifact)
          data_validation_artifact = data_validation.initiate_data_validation()

           #data transformation
          data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
          data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
          data_ingestion_artifact=data_ingestion_artifact)
          data_transformation_artifact = data_transformation.initiate_data_transformation()
        
           #model_trainer
          model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
          model_trainer= ModelTrainer(model_trainer_config= model_trainer_config,
          data_transformation_artifact= data_transformation_artifact)
          model_trainer_artifact = model_trainer.initiate_model_trainer()
          #model_evaluation
          model_evaluation_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
          model_evaluation=ModelEvaluation(model_eval_config=model_evaluation_config, 
          data_ingestion_artifact=data_ingestion_artifact,
          data_transformation_artifact=data_transformation_artifact,
          model_trainer_artifact=model_trainer_artifact) 
          data_eval_artifact = model_evaluation.initiate_model_evaluation()
          #model_pusher
          model_pusher_config  = config_entity.ModelPusherConfig(training_pipeline_config= training_pipeline_config)
          model_pusher = ModelPusher(model_pusher_config=model_pusher_config , 
          data_transformation_artifact=  data_transformation_artifact, 
          model_trainer_artifact= model_trainer_artifact)
          model_pusher_artifact = model_pusher.initiate_model_pusher()
     except Exception as e :
          raise SensorException(e, sys)

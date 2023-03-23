from census.logger import logging
from census.exception import SensorException
from census.utils import get_collection_as_dataframe
import sys,os
from census.entity.config_entity import DataIngestionConfig
from census.entity.config_entity import DataValidationConfig
from census.entity.config_entity import DataTransformationConfig
from census.entity import config_entity
from census.components.data_ingestion import DataIngestion
from census.components.data_validation import DataValidation
from census.components.data_transformation import DataTransformation

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
        

     except Exception as e :
          raise SensorException(e, sys)


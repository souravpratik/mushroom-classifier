from classifier.logger import logging
from classifier.exception import ClassifierException
from classifier.utils import get_collection_as_dataframe
import sys,os
from classifier.entity import config_entity
from classifier.components.data_ingestion import DataIngestion
from classifier.components.data_validation import DataValidation

training_pipeline_config = config_entity.TrainingPipelineConfig()

 #data ingestion
data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
print(data_ingestion_config.to_dict())
data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
#data validation
data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)

data_validation_artifact = data_validation.initiate_data_validation()
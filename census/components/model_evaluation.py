from census.predictor import ModelResolver
from census.entity import config_entity,artifact_entity
from census.exception import SensorException
from census.logger import logging
from census.utils import load_object
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas  as pd
import sys,os
from census.config import TARGET_COLUMN
class ModelEvaluation:

    def __init__(self,
        model_eval_config:config_entity.ModelEvaluationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact      
        ):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SensorException(e,sys)



    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #if saved model folder has model the we will compare 
            #which model is best trained or the model from saved model folder

            logging.info("if saved model folder has model the we will compare "
            "which model is best trained or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact


            #Finding location of transformer model and target encoder
            logging.info("Finding location of transformer model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()
            input_feature_encoder_path = self.model_resolver.get_latest_input_feature_encoder_path()
    

            logging.info("Previous trained objects of transformer, model and target encoder")

            #Previous trained  objects

            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)
            input_feature_encoder = load_object(file_path=input_feature_encoder_path)
            

            logging.info("Currently trained model objects")

            #Currently trained model objects

            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model  = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            current_input_feature_encoder = load_object(file_path=self.data_transformation_artifact.input_feature_encoder_path)


            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)  
            #for test_df
            test_df1 = test_df.copy()
            test_df1.drop('salary',axis=1,inplace=True)            
            cat_col=[]
            for i in test_df1.columns:
                if test_df1[i].dtype=='object':
                    cat_col.append(i)
            data_categorical = test_df1[cat_col]
            data_encoded = input_feature_encoder.transform(data_categorical)

            a = pd.DataFrame(data_encoded,columns=['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'country'])
            b = test_df[['age','hours-per-week','salary']]
            test_df = pd.concat([a,b],axis=1)     #test df

            
            target_df = test_df[TARGET_COLUMN]
            y_true =target_encoder.fit_transform(target_df)



            # accuracy using previous trained model
            
            input_feature_name = list(transformer.feature_names_in_)
            input_arr =transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)
            previous_model_score = accuracy_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")

            
            #for current df 
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path) 
            test_df1 = test_df.copy()
            test_df1.drop('salary',axis=1,inplace=True)
            cat_col=[]
            for i in test_df1.columns:
                if test_df1[i].dtype=='object':
                    cat_col.append(i)
            data_categorical = test_df1[cat_col]


            data_encoded = current_input_feature_encoder.transform(data_categorical)
            a = pd.DataFrame(data_encoded,columns=['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'country'])
            b = test_df[['age','hours-per-week','salary']]
            test_df = pd.concat([a,b],axis=1)     #test df            



            # accuracy using current trained model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr =current_transformer.transform(test_df[input_feature_name])
            y_pred = current_model.predict(input_arr)
            y_true =current_target_encoder.fit_transform(target_df)
            current_model_score = accuracy_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")
            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise SensorException(e,sys)

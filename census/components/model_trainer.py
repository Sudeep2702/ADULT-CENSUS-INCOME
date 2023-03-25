from census.entity import artifact_entity,config_entity
from census.exception import SensorException
from census.logger import logging
from typing import Optional
import os,sys 
from census import utils
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
class ModelTrainer:


    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)
        

    def fine_tune(self, X_train, y_train, X_test, y_test):
        try:
            # Define the hyperparameters to tune
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
                    
            # Instantiate the model
            rfc = RandomForestClassifier(random_state=42)
            
            # Instantiate the GridSearchCV object
            grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
            
            # Fit the GridSearchCV object to the data
            grid_search.fit(X_train, y_train)
            
            # Print the best hyperparameters found
            print("Best hyperparameters found: ", grid_search.best_params_)
            
            # Train the model with the best hyperparameters found
            best_rfc = RandomForestClassifier(**grid_search.best_params_, random_state=42)
            best_rfc.fit(X_train, y_train)
            
            # Evaluate the performance of the trained model on a separate test dataset
            test_accuracy = best_rfc.score(X_test, y_test)
            print("Test accuracy:", test_accuracy)
            
        except Exception as e:
            print(e)

    def train_model(self,x,y):
        try:
            rfc = RandomForestClassifier(**grid_search.best_params_, random_state=42)
            rfc.fit(x,y)
            return rfc
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("loading test and train array")
            train_arr = utils.load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path= self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            logging.info(f"finding the best parameters")
            best_params = self.fine_tune(X_train= x_train, y_train=y_train, X_test=x_test, y_test=y_test)
            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train)
            

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score  =f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score  =f1_score(y_true=y_test, y_pred=yhat_test)
            
            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")
            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            #save the grid search cv results
            logging.info(f"Saving cv results")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)            
            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_tuning_path, obj=best_params)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)


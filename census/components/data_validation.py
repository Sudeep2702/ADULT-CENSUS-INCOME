from census.entity import artifact_entity,config_entity
from census.exception import SensorException
from census.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os,sys 
import pandas as pd
from census import utils
import numpy as np
from census.config import TARGET_COLUMN
from sklearn.preprocessing import OrdinalEncoder


class DataValidation:


    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise SensorException(e, sys)

    

    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold

        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        =====================================================================================
        returns Pandas DataFrame if atleast a single column is available after missing columns drop else None
        """
        try:
            
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            #selecting column name which contains null
            logging.info(f"selecting column name which contains null above to {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            #return None no columns left
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise SensorException(e, sys)
    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
           
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base} is not available.]")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False
            return True
        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                #Null hypothesis is that both column data drawn from same distrubtion
                
                logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")
                same_distribution =ks_2samp(base_data,current_data)

                if same_distribution.pvalue>0.05:
                    #We are accepting null hypothesis
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                else:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }
                    #different distribution

            self.validation_error[report_key_name]=drift_report
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            
            base_df.replace({"na":np.NAN},inplace=True)
            logging.info(f"Replace na value in base df")
            #base_df has na as null
            logging.info(f"Drop null values colums from base df")
            base_df=self.drop_missing_values_columns(df=base_df,report_key_name="missing_values_within_base_dataset")

            # Clean the rows of object columns in the DataFrame
            for col in base_df.columns:
                if base_df[col].dtype == 'object':
                    base_df[col] = base_df[col].str.strip()
            # Find rows in base df with missing values and dropping them
            mask = base_df.applymap(lambda x: x == '?')
            base_df = base_df[~mask.any(axis=1)]

            #dropping unnecessary columns in base df
            base_df.drop(['fnlwgt','capital-gain','capital-loss'],axis=1,inplace=True) 
            base_df1 = base_df.copy()
            base_df1.drop('salary',axis=1,inplace = True)

            input_feature_encoder = OrdinalEncoder()

            #label encoding
            cat_col = []
            for i in base_df1.columns:
                if base_df1[i].dtype =='object':
                    cat_col.append(i)
            data_cat = base_df1[cat_col]

            data_encoded = input_feature_encoder.fit_transform(data_cat)
            a = pd.DataFrame(data_encoded,columns=['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'country'])
            b = base_df[['age','hours-per-week']]  
            base_df = pd.concat([a,b],axis=1) 
            #our label encoded base df is ready
            base_df.dropna(inplace=True)          


            #reading the train data frame
            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            # Clean the rows of object columns in the Train file
            for col in train_df.columns:
                if train_df[col].dtype == 'object':
                    train_df[col] = train_df[col].str.strip()
            train_df1 = train_df.copy()
            train_df1.drop('salary',axis=1,inplace = True)
            #getting categorical variables for label encoding
            cat_col1 = []
            for i in train_df1.columns:
                if train_df1[i].dtype== 'object':
                    cat_col1.append(i)
            data_cat1 = train_df1[cat_col1]

            data_encoded1 = input_feature_encoder.fit_transform(data_cat1)
            a = pd.DataFrame(data_encoded1,columns=['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'country'])
            b = train_df[['age','hours-per-week']]  
            train_df = pd.concat([a,b],axis=1) 
            #our label encoded train df is ready
            train_df.dropna(inplace=True)     

            #reading the test data frame
            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            # Clean the rows of object columns in the Train file
            for col in test_df.columns:
                if test_df[col].dtype == 'object':
                    test_df[col] = test_df[col].str.strip()
            test_df1 = test_df.copy()
            test_df1.drop('salary',axis=1,inplace = True)
            #getting categorical variables for label encoding
            cat_col2 = []   
            for i in test_df1.columns :
                if test_df1[i].dtype== 'object':
                    cat_col2.append(i)   
            data_cat2 = test_df1[cat_col2] 
            data_encoded2 = input_feature_encoder.fit_transform(data_cat2)
            a = pd.DataFrame(data_encoded2,columns=['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'country'])
            b = test_df[['age','hours-per-week']]  
            test_df = pd.concat([a,b],axis=1) 
            #our label encoded train df is ready
            train_df.dropna(inplace=True)     

            logging.info(f"Drop null values colums from train df")
            train_df = self.drop_missing_values_columns(df=train_df,report_key_name="missing_values_within_train_dataset")
            logging.info(f"Drop null values colums from test df")
            test_df = self.drop_missing_values_columns(df=test_df,report_key_name="missing_values_within_test_dataset")
            
            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)


            logging.info(f"Is all required columns present in train df")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df,report_key_name="missing_columns_within_train_dataset")
            logging.info(f"Is all required columns present in test df")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df,report_key_name="missing_columns_within_test_dataset")

            if train_df_columns_status:
                logging.info(f"As all column are available in train df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df,report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                logging.info(f"As all column are available in test df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df,report_key_name="data_drift_within_test_dataset")

            #write the report
            logging.info("Write reprt in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path,)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    


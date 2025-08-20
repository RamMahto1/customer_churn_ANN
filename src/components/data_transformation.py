from src.logger import logging
from src.exception import CustomException
import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,classification_report,recall_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils import saved_obj

class DataTransformationConfig:
    preprocessor_path_obj:str = os.path.join("artifacts/preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_data_transformation_obj(self):
        try:
            numerical_feature = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend',
                                 'Last Interaction']
            
            categorical_feature = ['Gender','Subscription Type','Contract Length']
            
            num_pipeline = Pipeline(
                steps=[
                    ("num",SimpleImputer(strategy="mean")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("cat",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encorder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipe",num_pipeline,numerical_feature),
                    ("cat_pipe",cat_pipeline,categorical_feature)
                ]
            )
            
            logging.info(f"numerical features:{numerical_feature}")
            logging.info(f"categorical feature:{categorical_feature}")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read data as data frame")
            logging.info(f"obtaining preprocessor path obj")
            preprocessor_obj = self.get_data_transformation_obj()
            
            target_columns = ['Churn']
            
            input_feature_train_df = train_df.drop(columns = target_columns)
            target_feature_train_df = train_df[target_columns]
            
            input_feature_test_df = test_df.drop(columns = target_columns)
            target_feature_test_df = test_df[target_columns]
            
            logging.info("applying preprocessor path on training and testing dataset")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df.to_numpy()]
            
            saved_obj(
                file_path=self.data_transformation_config.preprocessor_path_obj,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_path_obj
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)
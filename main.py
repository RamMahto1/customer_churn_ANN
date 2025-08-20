from src.logger import logging
from src.exception import CustomException
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


def main():
    try:
        # step: 1 Data Ingestion
        data_ingestion = DataIngestion()
        train_data,test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        
        # step: 2 Data Transformation
        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformer(train_data,test_data)
        
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
    
    
    
if __name__=="__main__":
    main()
# logging.info(f"logging has started")

# try:
#     a = 1/0
#     logging.info("1 divided by zero")
# except Exception as e:
#     raise CustomException(e,sys)
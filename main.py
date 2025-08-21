from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def main():
    try:
        logging.info("Pipeline started.")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformer(train_data, test_data)
        logging.info("Data transformation completed.")

        # Step 3: Model Trainer
        X_train = train_arr[:, :-1]  # features
        y_train = train_arr[:, -1]   # target
        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        model_trainer = ModelTrainer()
        acc = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        logging.info(f"Final Model Accuracy: {acc}")

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()

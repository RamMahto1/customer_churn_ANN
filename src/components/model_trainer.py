from src.logger import logging
from src.exception import CustomException
import os, sys
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score
import numpy as np

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_model(self, input_dim, num_classes):
        try:
            model = Sequential()
            model.add(Dense(64, activation='relu', input_dim=input_dim))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dropout(0.2))

            if num_classes == 2:  # Binary classification
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
            else:  # Multi-class classification
                model.add(Dense(num_classes, activation='softmax'))
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

            return model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            input_dim = X_train.shape[1]
            num_classes = len(np.unique(y_train))

            model = self.build_model(input_dim, num_classes)

            early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

            model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      epochs=50,
                      batch_size=32,
                      callbacks=[early_stop],
                      verbose=1)

            model.save(self.config.model_path)
            logging.info(f"Model saved at {self.config.model_path}")

            if num_classes == 2:
                y_pred = (model.predict(X_test) > 0.5).astype("int32")
            else:
                y_pred = np.argmax(model.predict(X_test), axis=1)

            acc = accuracy_score(y_test, y_pred)
            logging.info(f"Model Accuracy: {acc}")

            return acc

        except Exception as e:
            raise CustomException(e, sys)

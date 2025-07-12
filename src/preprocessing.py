import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

import logging
import os


logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("preprocessing")
logger.setLevel('DEBUG')

console_logger = logging.StreamHandler()
console_logger.setLevel('DEBUG')

log_path = os.path.join(logs_dir, 'preprocessing.log')
file_logger = logging.FileHandler(log_path)
file_logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

console_logger.setFormatter(formatter)
file_logger.setFormatter(formatter)

logger.addHandler(console_logger)
logger.addHandler(file_logger)

def preprocessing(path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''this function process data'''
    try:
        df = pd.read_csv(path)
        logger.debug("data loaded successfully")

        df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
        logger.debug("unwanted columns removed successfully")

        y = df['diagnosis']
        X = df.drop(columns=['diagnosis']).copy()
        logger.debug("features and labels separeted.")

        tf = PowerTransformer()
        le = LabelEncoder()
        X =  tf.fit_transform(X)
        y = le.fit_transform(y)
        logger.debug("features and labels encoded")

        return X, y
    except Exception as e:
        logger.error("ERROR: ",e)
        raise


def main():
    try:
        data_path = 'data/raw_data/Cancer_Data.csv'
        x_train, y_train = preprocessing(data_path)
        logger.debug("data processe is successfull")

        clean_data_path = 'data/clean_data'
        os.makedirs(clean_data_path, exist_ok=True)
        x_train_name = os.path.join(clean_data_path, 'x_train.npy')
        y_train_name = os.path.join(clean_data_path, 'y_train.npy')
        np.save(x_train_name, x_train)
        np.save(y_train_name, y_train)
        logger.debug("x_train and y_trai saved")
    except Exception as e:
        logger.error("ERROR: ",e)
        raise

    

if __name__ == "__main__":
    main()

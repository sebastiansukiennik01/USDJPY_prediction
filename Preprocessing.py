"""
File with Preprocessing class, functionality for loading and initial transformation of data
"""
import os
import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import normalize
from tensorflow.data import Dataset


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option("display.width", None)



class Preprocessing(object):

    def __init__(self):
        self.data = None
        self.train_x: pd.DataFrame = pd.DataFrame()
        self.validation_x: pd.DataFrame = pd.DataFrame()
        self.test_x: pd.DataFrame = pd.DataFrame()

        self.train_y: pd.DataFrame = pd.DataFrame()
        self.validation_y: pd.DataFrame = pd.DataFrame()
        self.test_y: pd.DataFrame = pd.DataFrame()

        self.train = None
        self.validation = None
        self.test = None

        self.labels: list = None
        self.sets = None

    def loadDataFromCache(self, fileName: str, labels: list):
        """
        Assumes that file is csv and is located in ./data/ folder, loads the data.
        :param fileName: file path to data
        :return: self
        """
        self.labels = labels if isinstance(labels, list) else [labels]
        path = os.path.join('./data', f'{fileName}.csv')
        self.data = pd.read_csv(path, index_col=[0]).reset_index(drop=True)

        return self

    def divideTrainTest(self, testSize: float = 0.1, validationSize: float = 0.1):
        """
        Divides dataset to train, test (and validation if provided). Assign each set as attribute.
        :return: self
        """
        trainEnd = int(self.data.shape[0] * (1 - validationSize - testSize))
        validationEnd = int(self.data.shape[0] * (1 - testSize))

        train = self.data.iloc[:trainEnd, :]
        val = self.data.iloc[:validationEnd, :]
        test = self.data.iloc[validationEnd:, :]

        self.train_x = train.drop(columns=self.labels)
        self.validation_x = val.drop(columns=self.labels)
        self.test_x = test.drop(columns=self.labels)

        self.train_y = train.loc[:, self.labels]
        self.validation_y = val.loc[:, self.labels]
        self.test_y = test.loc[:, self.labels]

        return self

    def standarize(self):
        """
        Normalizes all labels.
        :return: self
        """
        if self.labels:
            for d in [self.train_x, self.validation_x, self.test_x]:
                temp_stand = (d - d.mean()) / d.std()

                # remove Nan columns (const values when standardized give Nan)
                temp_stand = temp_stand.dropna(axis=1)
                mask = list(temp_stand.columns)
                d[mask] = temp_stand
        else:
            temp = self.data.drop(columns=[self.labels])
            temp_stand = (temp - temp.mean()) / temp.std()
            temp_stand = temp_stand.dropna(axis=1)
            mask = list(temp_stand.columns)
            self.data[mask] = normalize(temp)

        return self

    def createDataset(self, batchSize: int = 32, prefetch: bool = True):
        """
        Creates the dataset from tensor slices and assign it to attribute.
        :param batchSize: batch size for dataset
        :param prefetch: if True then model will prepare next batch during training previous badge
        :return:
        """
        p = int(prefetch)
        # self.train = Dataset.from_tensor_slices((self.train_x, self.train_y))
        # self.validation = Dataset.from_tensor_slices((self.validation_x, self.validation_y))
        # self.test = Dataset.from_tensor_slices((self.test_x, self.test_y))

        self.train_x = tf.convert_to_tensor(self.train_x)
        self.train_y = tf.convert_to_tensor(self.train_y)

        self.validation_x = tf.convert_to_tensor(self.validation_x)
        self.validation_y = tf.convert_to_tensor(self.validation_y)

        self.test_x = tf.convert_to_tensor(self.test_x)
        self.test_y = tf.convert_to_tensor(self.test_y)

        return self.train_x, self.train_y, self.validation_x, self.validation_y, self.test_x, self.test_y

    def getDataset(self) -> Dataset:
        """
        Returns dataset as 3 elements train, validation and test dataset
        :return: 3 tuples of two.
        """
        return self.train, self.validation, self.test

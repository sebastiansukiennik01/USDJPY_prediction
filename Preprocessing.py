"""
File with Preprocessing class, functionality for loading and initial transformation of data
"""
import os
import pandas as pd
import tensorflow as tf


class Preprocessing(object):

    def __init__(self):
        self.data = None

    def loadDataFromCache(self, fileName: str):
        """
        Assumes that file is csv and is located in ./data/ folder, loads the data.
        :param fileName: file path to data
        :return: self
        """
        path = os.path.join('data', f'{fileName}.csv')
        self.data = pd.read_csv(path)

        return self

    def divideTrainTest(self, test: float = 0.2, validation: float = 0.2):
        """
        Divides dataset to train, test (and validation if provided). Assign each set as attribute.
        :return:
        """
        pass

    def shuffle(self):
        """
        Shuffle the dataset, if already divided into test, validation, shuffle them and leave test as is.
        :return:
        """
        pass

    def normalize(self):
        """
        Normalizes all features.
        :return:
        """
        pass

    def setBatch(self, batchSize: int = 32):
        """
        Sets batches on dataset.
        :param batchSize: size of a batch
        :return:
        """
        pass

    def prefetch(self):
        """
        Add prefetch in order to prepare next batches when learning.
        :return:
        """
        pass

    def getDataset(self) -> tuple:
        """
        Returns dataset as 6 elements (train_x, train_y), (validate_x, validate_y), (test_x, test_y)
        :return: 3 tuples of two.
        """
        pass

    def test_tf(self):
        pass

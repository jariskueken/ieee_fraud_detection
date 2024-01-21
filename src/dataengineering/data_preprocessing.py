from typing import Any

import pandas as pd
import numpy as np
import logging

from util import file_util

from sklearn.preprocessing import StandardScaler

TYPE_TRAIN = 'train'
TYPE_TEST = 'test'


class DataPreprocessor():
    """
    A simple class that automates data preprocessing steps for given dataset,
    starts by checking if the provided data file exists and then reads the
    data. Preprocessed the data after the given steps.
    """
    def __init__(self,
                 train_data_path: str,
                 test_data_path: str,
                 target: str = '',
                 predict: bool = False,
                 verbose: bool = False):

        self.verbose = verbose
        self.target = target
        self.features: list[str] = []

        # the prepared data
        self.train_data_X: np.ndarray
        self.train_data_y: np.ndarray
        self.test_data_X: np.ndarray

        # build the data, build always for train data and only for test data
        # if we want to predict
        self.__build_from_file(train_data_path, TYPE_TRAIN)
        if predict:
            self.__build_from_file(test_data_path, TYPE_TEST)

    def __read_data(self, f: str) -> pd.DataFrame:
        """
        A private method to read the data from the csv file
        """
        file_path_exists_error = file_util.check_file_path_exists(f)
        if self.verbose:
            logging.debug('check if file exists and is of type csv file...')
        # first check if the filepath points to a valid file
        if file_path_exists_error is not None:
            logging.error(f'received invalid file path: \
{file_path_exists_error}')
        # now check if the file is a csv file
        file_is_csv_error = file_util.check_file_is_csv(f)
        if file_is_csv_error is not None:
            logging.error(f'received non csv path: {file_is_csv_error}')
        # now that we checked everything we can read the data
        if self.verbose:
            logging.debug('check completed, reading data file now...')

        return pd.read_csv(f, index_col=None)

    def __get_features(self,
                       data: pd.DataFrame
                       ) -> None:
        """
        A private method that gets a target feature and selects all features
        from the provided dataset from there on. Writes the features list into
        'self.features'.
        """

        features = data.columns.values.tolist()
        # store the features as str names
        features = [str(feature) for feature in features]

        # drop the target from the list of features as we don't want to train
        # with the target
        if self.target != '':
            if self.target in features:
                logging.debug(f'found target {self.target} in list of \
features. Deleting...')
                features.remove(self.target)

        self.features = features

    def __build_from_file(self,
                          path: str,
                          type: str) -> None:
        """
        builds the dataframes for the datasets provided as paths on creation
        of this object
        """

        # start by reading the data
        logging.debug(f'reading data from {path}')
        data = self.__read_data(path)

        # extract the features if they don't already exist
        if not self.features:
            logging.debug('features are empty getting now..')
            self.__get_features(data)

        # build the dataframes
        if type == TYPE_TRAIN:
            logging.debug('building X and Y vectors for training data')
            self.train_data_X = data[self.features].to_numpy()
            self.train_data_y = data[self.target].to_numpy()

        elif type == TYPE_TEST:
            logging.debug('building X vector for test data')
            self.test_data_X = data[self.features].to_numpy()

        del data

    def preprocess_data(self) -> None:
        """
        Preprocesses data with every necessary steps

        Returns the X and Y values of the dataset as tuple of two ndarrays
        where the tuple is of shape (X,y)

        Paramters:
            - 'features' is a list of any type but must match the type of the
                given dataset in the dont_overfit_ii case its strings.
                It contains the names of all the target collumns in the
                dataset that should be used
            - 'target' is a string which contains the name of the target value
                in the dataset must also match the type of the collumn name
            - 'f' contains the file path to the data file
            - 'scale' a boolean to determine if we want to standardscale the
                data or not. Default is true but for some datasets it might
                be better to not prescale the data.

        While preprocessing it does the following steps
        1. Select only the feature collumns that are wanted by the user
        2. Standard scale the features
        3. Return the standard scaled data with only the selected features
        4. Return the data as Dataframe again
        """

        # FIXME: can't fit on test data needs scaling factors of training
        # data and then just run transform on test data
        scaler = StandardScaler()
        logging.debug(f'scaling train data of shape: \n Rows: \
{self.train_data_X.shape[0]} \n Collumns: {self.train_data_X.shape[1]}')
        self.train_data_X = scaler.fit_transform(self.train_data_X)
        logging.debug(f'scaling test data of shape: \n Rows: \
{self.test_data_X.shape[0]} \n Collumns: {self.test_data_X.shape[1]}')
        self.test_data_X = scaler.transform(self.test_data_X)

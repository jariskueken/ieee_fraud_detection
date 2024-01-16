from typing import Any

import pandas as pd
import numpy as np
import logging

from util import file_util

from sklearn.preprocessing import StandardScaler


class DataPreprocessor():
    """
    A simple class that automates data preprocessing steps for given dataset,
    starts by checking if the provided data file exists and then reads the
    data. Preprocessed the data after the given steps.
    """
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.data = pd.DataFrame()
        self.features: list[str] = []

    def __read_data(self, f: str) -> None:
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

        self.data = pd.read_csv(f, index_col=None)

    def __get_features(self,
                       target: str = '') -> None:
        """
        A private method that gets a target feature and selects all features
        from the provided dataset from there on. Writes the features list into
        'self.features'.
        """

        features = self.data.columns.values.tolist()
        # store the features as str names
        features = [str(feature) for feature in features]

        # drop the target from the list of features as we don't want to train
        # with the target
        if target != '':
            if target in features:
                logging.debug(f'found target {target} in list of features. \
Deleting...')
                features.remove(target)

        self.features = features

    def preprocess_data(self,
                        f: str,
                        target: str = '',
                        scale: bool = True,
                        ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray,
                                                                   None]:
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

        self.__read_data(f)
        self.__get_features(target)

        # select only the provided features and also the targets
        targets = None
        # only select targets if parameter is provided
        if target != '':
            targets = self.data[target].to_numpy()
            if self.verbose:
                logging.debug(f'targt vector is of shape {targets.shape}')

        if scale:
            #FIXME: can't fit on test data needs scaling factors of training data and then just run transform on test data
            scaler = StandardScaler()
            prepared_features = scaler.fit_transform(self.data[self.features])
            logging.debug(f'scaled feature vector is of shape: \
{prepared_features.shape}.')
        else:
            prepared_features = self.data[self.features].to_numpy()
            logging.debug(f'prepared feature vector is of shape: \
{prepared_features.shape}. Did not standardscale the features as scale \
parameter was False')
        return tuple([prepared_features, targets])  # type: ignore

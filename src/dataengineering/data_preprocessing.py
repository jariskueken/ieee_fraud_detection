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

    def _read_data(self, f: str) -> pd.DataFrame:
        """
        A private method to read the data from the csv file
        """
        file_path_exists_error = file_util.check_file_path_exists(f)
        if self.verbose:
            logging.error('check if file exists and is of type csv file...')
        # first check if the filepath points to a valid file
        if file_path_exists_error is not None:
            logging.error(f'received invalid file path: {file_path_exists_error}')
        # now check if the file is a csv file
        file_is_csv_error = file_util.check_file_is_csv(f)
        if file_is_csv_error is not None:
            logging.error(f'received non csv path: {file_is_csv_error}')
        # now that we checked everything we can read the data
        if self.verbose:
            logging.error('check completed, reading data file now...')
        return pd.read_csv(f)

    def preprocess_data(self,
                        f: str,
                        features: list[Any],
                        target: Any = None
                        ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, None]:
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

        While preprocessing it does the following steps
        1. Select only the feature collumns that are wanted by the user
        2. Standard scale the features
        3. Return the standard scaled data with only the selected features
        4. Return the data as Dataframe again
        """
        data = self._read_data(f)
        scaler = StandardScaler()

        # select only the provided features and also the targets
        features = data[features]

        targets = None
        # only select targets if parameter is provided
        if target is not None:
            targets = data[target].to_numpy()
            if self.verbose:
                logging.debug(f'targt vector is of shape {targets.shape}')

        scaled_features = scaler.fit_transform(features)
        if self.verbose:
            logging.debug(f'scaled feature vector is of shape: {scaled_features.shape}.')

        return tuple([scaled_features, targets])  # type: ignore

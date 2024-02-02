from typing import Any
from datetime import datetime
import os
import logging

from util import file_util

import numpy as np
import pandas as pd

from model.model_builder import ModelBuilder


class Predictor:
    """
    A class to make a prediction on a given dataset using ml classifers
    """

    def __init__(self,
                 train_data_X: np.ndarray,
                 train_data_y: np.ndarray,
                 test_data_X: np.ndarray,
                 dataset_identifier: str,
                 model_builder: ModelBuilder,
                 verbose: bool
                 ):
        """
        The predictor always needs the training data as well as the test data
        to be able to train the given model on the data and make a prediciton
        from that. On initialization this class gets the features as well as
        the targets of the training data as well as the features from the test
        data as ndarrays.
        """
        self.verbose = verbose

        self.train_data_X = train_data_X
        self.train_data_y = train_data_y
        self.test_data_X = test_data_X
        self.dataset_identifier = dataset_identifier
        self.model_builder = model_builder

    def _store_prediction(self,
                          prediction: np.ndarray,
                          clf_identifier: str,
                          submission_template_path: str = None,
                          submission_dir: str = None,
                          target: str = None,
                          ) -> str:
        """
        A private method to store the prediciton into a submission csv file.
        Creates a new submission csv file for every time this method is called
        so we assume that each time we call this method we made a new
        prediction

        Return:
            - returns the path to the submssion file
        """
        # if we want to store the prediction we need to check wether we
        # have a submission_template as well as a submission_dir path set
        # otherwise we are unable to store the template and should quit
        if (submission_template_path is None or
                submission_dir is None or
                target is None):
            logging.error(f'Received {submission_template_path} as template path \
                    and {submission_dir} as submission directory as well as \
                        {target} as target column name. Unable to \
                            store submission this way. Quitting...')
            return ""
        # TODO: refactor this code to reduce code duplication as it is the
        # same code as in data_preprocessing
        # check if the csv file exists and if it is a csv file
        file_path_exists_error = file_util.check_file_path_exists(
            submission_template_path)
        if self.verbose:
            logging.debug('check if submission file exists and is of type csv \
file...')
        # first check if the filepath points to a valid file
        if file_path_exists_error is not None:
            logging.error(f'received invalid file path: \
{file_path_exists_error}')
        # now check if the file is a csv file
        file_is_csv_error = file_util.check_file_is_csv(
            submission_template_path)
        if file_is_csv_error is not None:
            logging.error(f'received non csv path: {file_is_csv_error}')
        # TODO: check if the submission directory exists

        # the actual logic to write to the submission file
        submission = pd.read_csv(submission_template_path, index_col=None)

        # check if the target column and the prediciton are of equal shape
        if submission[target].shape[0] != len(prediction):
            logging.error(f'submission target column has size \
{submission[target].shape[0]} and prediction has size {prediction}. Sizes \
have to be equal')
            return ""

        if self.verbose:
            logging.debug('Creating submission file...')
        # overwrite the target collumn, just pass the proba for target = 1 to
        # match roc_auc format
        submission[target] = [prediction[i][1] for i in range(len(prediction))]

        # Create the submissions for each day folder if it doesn't exist
        date = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(os.path.join(submission_dir, date)):
            os.makedirs(os.path.join(submission_dir, date))

        # generate the output file name, format the date aftetr ISO8601
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        output_file = f'submission_{timestamp}_{clf_identifier}_\
{self.dataset_identifier}.csv'
        output_path = os.path.join(submission_dir, date, output_file)

        # store the submission file in the correct path
        submission.to_csv(output_path, index=False)

        if self.verbose:
            logging.debug(f'Successfully stored submission at {output_path}')

        return output_path

    def train(self,
              clf: Any,
              clf_identifier: str
              ) -> Any:
        """
        A method to train a given model on the train dataset that was provided
        on creation of this class. Can take parameters to refine the training
        process currently only default training on the entire dataset.
        NOTE: currently this uses a basic approach of using a blank model and
        training this model on the entire training dataset. May be changed
        later to improve model scores using ensembles.

        Paramters:
            - 'clf' the classifier to train.
            - 'clf_identifier' the name of the classifier that is beeing
                trained at the moment for better logging output.
            - 'store' bool to define if we want to store the trained model
        """

        # fit the model on the training data
        if self.verbose:
            logging.debug(f'Fitting training data on {clf_identifier} \
classifier...')

        clf.fit(self.train_data_X, self.train_data_y)

        self.model_builder.export_model(clf,
                                        clf_identifier,
                                        self.dataset_identifier)
        return clf

    def predict(self,
                clf: Any,
                clf_identifier: str,
                store_model: bool = False,
                store_prediction: bool = False,
                submission_template_path: str = None,
                submission_dir: str = None,
                target: str = None,
                pretrained: bool = False,
                ) -> tuple[np.ndarray, str]:
        """
        A method to make a prediciton on a given dataset. The Predictor class
        receives the data on model creation hence why there is no need to read
        data beforehand. The classifier is trained by the train method on the
        test dataset with the provided paramteres/settings.

        Parameters:
            - 'clf' the classifier to use to make a prediction.
            - 'clf_identifier' a unique identifier for the classifier that is
                tested currently to differntiate between the clfs afterwards
            - 'store_model' bool to define if we want to store the model after
                training.
            - 'store_prediciton' bool to define if we want to write and store
                the prediciton into a submission file.
            - 'submission_template_path' a path to the template submission
                file. Must be a csv file
            - 'submission_dir' a path to the directory where the submission
                should be stored
            - 'target' the name of the column in the submission file where the
                prediciton should be written to. #TODO: maybe it is better to
                default this to 'target'
            - 'pretrained' defines if we use a stored model that was already
                pretrained so we don't need to train it again. Applys also if
                we use an ensemble as this model is also pretrained.

        Return:
            - the ndarray containing the predictions if we just want to get
                the predicitons
            - a tuple containing the prediciton as well as the path to the
                submission file if we want to store the prediciton
        """

        # only train the model if we dont have an ensemble
        clf_trained = clf
        if not pretrained:
            # overwrite the trained clf if we train it again
            clf_trained = self.train(clf,
                                     clf_identifier)

        # make a prediction, the predictions in the prediciton array should be
        # ordered in the same way as the order of the test samples in
        # self.test_data_X
        try:
            prediction = clf_trained.predict_proba(self.test_data_X)
        except Exception as e:
            # if we get an exception just return an array of zeros that has
            # the same size as the target array
            logging.error(f' got an exception trying to predict with \
{clf_identifier}: {e}')
            # build a 2d array to match the dimensionalty of a predict proba
            # array
            prediction = np.zeros((self.train_data_y.shape[0], 2))
        submission_file = ""
        if store_prediction:
            # store the prediction
            submission_file = self._store_prediction(prediction,
                                                     clf_identifier,
                                                     submission_template_path,
                                                     submission_dir,
                                                     target)
        logging.info(f'finished predicting on {clf_identifier}')

        return tuple([prediction, submission_file])



from typing import Any
import os
from datetime import datetime
import logging
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, StackingClassifier

import pickle

from util import file_util


class ModelBuilder:
    """
    A simple class to handle model building for ML models. Can ex- and import
    models as well as build ensemble models
    """
    def __init__(self,
                 train_data_X: np.ndarray,
                 train_data_y: np.ndarray,
                 verbose: bool):
        self.verbose = verbose

        self.train_data_X = train_data_X
        self.train_data_y = train_data_y

    def export_model(self,
                     clf: Any,
                     clf_identifier: str,
                     dir: str
                     ) -> str:
        """
        A method to export a given model. The model should be trained
        beforehand and will be exported using the pickle module.

        Parameters:
            - 'clf' a classifier, should be already trained.
            - 'clf_identifier' a unique string containing a identifier for the
                given model. Should contain all relevant information about the
                model as it will be part of the filename for the stored model.
            - 'dir' the relative path of the directory where the model should be
                stored
        Return:
            - a string containing the path to the file where the model is
                stored.
        """
        # check if the directory exists, if not create
        date = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(os.path.join(dir, date)):
            if self.verbose:
                logging.debug("model directory does not exists, creating now...")
            os.makedirs(os.path.join(dir, date))

        # build the correct identifier
        # generate the output file name, format the date after ISO8601
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        filepath = f'{timestamp}_{clf_identifier}.pkl'
        output_path = os.path.join(dir, date, filepath)

        # write the model to the file
        with open(output_path, 'wb'):
            if self.verbose:
                logging.debug(f'storing model at {output_path}')
            pickle.dump(clf)

        return output_path

    def import_model(self,
                     path: str
                     ) -> Any:
        """
        TODO: write documentation
        """
        file_path_exists_error = file_util.check_file_path_exists(path)
        if file_path_exists_error is not None:
            logging.error(f'file at {path} does not exist')
        # check if the file is a pickle file
        file_is_pickle_error = file_util.check_file_is_pickle(path)
        if file_is_pickle_error is not None:
            logging.error(f'file path does not point to a pickle\
                  file: {file_is_pickle_error}')
        # if path exists load the model and return
        with open(path, 'rb'):
            clf = pickle.load()

        return clf

    def build_stacking_ensemble(self,
                             clfs: list[Any],
                             clf_identifiers: list[str],
                             use_all_features: bool = False,
                             splits: int = 10,
                             random_state: int = 1
                             ) -> tuple[str, Any]:
        """
        builds a stacking model from a classifier where it uses k folds to
        make predictions and stacks these predictions into one final model.
        
        Paramters:
            - 'clf' the classifier we want to use
            - 'clf_identifier' the identifier of the classifier we want to use
        
        Return:
            - a tuple containing the the idetnifier of the new classifier and
                the new clf itself
        """
        # we need a train_meta set for later training where we store the
        # predictions our stratified models make. Then we add this column to
        # our train set later on as a new collumn which is used for training
        # the stacking model afterwards
        train_meta_y = []
        train_meta_x = []

        # we use stratified k fold
        skf = StratifiedKFold(n_splits=splits,
                              shuffle=True,
                              random_state=random_state)
        skf.get_n_splits(self.train_data_X, self.train_data_y)

        # run for each classifier
        for clf, name in zip(clfs, clf_identifiers):
            # set a new train meta for each classifier
            current_meta_y = np.zeros(self.train_data_y.shape, dtype=float)

            logging.debug(f'running stratified k fold on base model {name}')
            # iterate over every split
            for i, (train_idx, test_idx) in enumerate(skf.split(self.train_data_X, self.train_data_y)):
                logging.debug(f'currently building stacking meta model on {i}.st of {splits} splits using {name} classifier')
                # extract the train and test fold
                x_train_fold, x_test_fold = self.train_data_X[train_idx], self.train_data_X[test_idx]
                y_train_fold, _ = self.train_data_y[train_idx], self.train_data_y[test_idx]

                # make a prediction for the current split and store result, calc
                # the performance of the clf on the current split afterwards
                clf.fit(x_train_fold, y_train_fold)

                # predict on the test fold
                logging.debug(f'predicting on test fold {i} using {name} classifier')
                y_split = clf.predict(x_test_fold)
                # our predictions are in order of the indices of the test fold
                # split. We can now change all of the indices that we predicted in
                # our meta training set
                for i in range(len(test_idx)):
                    # test idx and y_split have the same length
                    # we want to replace the element in current_meta_y at the
                    # current position of test_idx with the resulting prediction
                    # value which is at the same position as we are in the list of
                    # idx
                    idx_to_update = test_idx[i]
                    current_meta_y[idx_to_update] = y_split[i]
            # add the current meta for each individual model to train_meta_y
            train_meta_y.append(current_meta_y)
            logging.debug(f'finished creating train meta set using k fold cv,\
got new collumn of shape {current_meta_y.shape}')

        # differntiate if we want to use all features or only our new features
        # to train our stacking model
        if use_all_features:
            logging.debug('Using all features for stacking classifier')
            # concatenate the results from each model with the original
            # features
            new_features = np.array(train_meta_y).T

            for i in range(len(self.train_data_X)):
                new_row = list(self.train_data_X[i]) + list(new_features[i])
                train_meta_x.append(new_row)
        else:
            logging.debug('Using only newly created features for stacking\
classifier')
            # if not we create our meta set as only new features representing
            # our prediciton from before stashing the old features completely
            train_meta_x = [list(elem) for elem in zip(*train_meta_y)]
        # turn back to np array
        train_meta_x = np.array(train_meta_x)

        # create the stacking classifier
        estimators = list(zip(clf_identifiers, clfs))
        logging.debug(f'creating stacking classifier using {estimators}')
        sclf = StackingClassifier(estimators)

        # fit the stacking classifier on the new meta set
        sclf = sclf.fit(train_meta_x, self.train_data_y)
        return sclf

    def build_voting_ensemble(self,
                              clfs: list[Any],
                              clf_identifiers: list[str],
                              ) -> Any:
        """
        TODO: write documentaion for this case where we use voting classifier
        ref: https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

        """
        # check if clf and identifiers are of same length
        if len(clfs) != len(clf_identifiers):
            logging.error('got unequal ammount of clfs and identifiers')
            return None
        # for each clf build a tuple of (clf_identifier, clf)
        estimators = []
        for i, clf in enumerate(clfs):
            current_clf = (clf_identifiers[i], clf)
            estimators.append(current_clf)

        eclf = VotingClassifier(
            estimators=estimators,
            voting='hard')

        return eclf

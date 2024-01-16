from typing import Any
import logging


from statistics import mean
import numpy as np
from util.decorators import timeit

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class Evaluator():
    """
    A class that evaluates prediction models on a given data set
    """
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray | None,
                 verbose: bool) -> None:
        """
        The constructor for the class. Must be initalized with the x and y data
        we want to predict on as well as the verbose attribute
        Paramteres:
            - 'x' the feature data of our dataset
            - 'y' the target values of out dataset, can also be empty if we
                want to make a prediction on a test set
        """
        self.x = x
        self.y = y
        self.verbose = verbose

    @timeit
    def cross_validate_model(self,
                             clf: Any,
                             x: np.ndarray,
                             y: np.ndarray,
                             splits: int = 10,
                             random_state: int = 1,
                             ) -> list[float]:
        """
        This method runs stratified k-fold cross validation on the provided
        data and trains the given model on the data using the split datasets.
        Parameters:
            - 'clf' is the classifier we want to test. This is of any type but
                should have a fit method every time.
            - 'x' a vector of all samples with the selected features
            - 'y' a vector of all target values with all features.
            - 'splits' the number of splits k we want to do in our k-fold CV
            - 'random_state' is an optional int that we pass to the stratified
                k-fold CV if we want to keep the splits consistend for our
                model evaluation. If we dont set it, it will be None and we
                get new splits every time. NOTE: if we want to compare model
                we should set this parameter to get the same split for every
                model

        Return:
            - a list of floats containing the average scores for each fold
                prediction of the model
        """

        # split the training set into k splits. We shuffle every time, and
        # random_state is optional depending on our use case
        skf = StratifiedKFold(n_splits=splits,
                              shuffle=True,
                              random_state=random_state)
        skf.get_n_splits(x, y)
        scores = []
        if self.verbose:
            logging.debug(f'evaluating model {clf} on {splits} splits using \
stratified k-cross validation')

        # iterate over every split
        for i, (train_idx, test_idx) in enumerate(skf.split(x, y)):
            logging.debug(f'running {i}th split for {clf}')

            # handle with try except to prevent entire testing process to fail
            # because of one classifier
            try:
                # extract the train and test fold
                x_train_fold, x_test_fold = x[train_idx], x[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]

                # make a prediction for the current split and store result,
                # calc the performance of the clf on the current split
                # afterwards
                clf.fit(x_train_fold, y_train_fold)

                # handle classifiers that are not able to predict probabilities
                y_test_fold_predict = clf.predict_proba(x_test_fold)

                counter = 0
                for prediction in y_test_fold_predict:
                    if prediction[1] >= 0.5:
                        counter += 1

                fraud = [idx for idx in range(len(y_test_fold)) if y_test_fold[idx] == 0]
                fraud_probas = y_test_fold_predict[fraud]
                # logging.debug(f'predicted fraud in {(len(y_test_fold_predict)/counter) * 100}% of cases')
                # use the probabilities for label 1 for roc curve as we use TP
                # and FP for the curve
                score = roc_auc_score(y_test_fold, y_test_fold_predict[:, 1])
                scores.append(score)
            except Exception as e:
                logging.debug(f'got an error for {clf} breaking...: {e}')
                scores.append(0.0)
                break

        return scores

    @timeit
    def evaluate_model(self,
                       clfs: list[Any],

                       clf_names: list[str]
                       ) -> dict[str, tuple[float, list[float]]]:
        """
        Evaluates the model given as paramters using stratified
        k-cross-validation. Can predict one or multiple models sequentually
        defined by the paramteres provided. Stores the scores
        into a dictionary, mapping the classifier name to a tuple containing
        the list of scores it acchieved on the stratified-k-CV and the average
        score of the classifiers it acchieved in k rounds of stratified CV.

        Parameters:
            - 'clfs' a list of classifiers to test, must be a list even if we
                only test one classifier to simplify the code
            - 'clf_name' a list of strings with the corresponding human
                readable names of each classifier to better map the score
                to a classifier

        Return:
            - a hashmap of type -> dict: (clf_name: (avg_score, [scores])
        """
        # lets throw an error if we dont have equal ammount of names and clfs
        # as we would be unable to map them correctly
        if len(clfs) != len(clf_names):
            logging.error(f'Received {len(clfs)} classifiers and \
{len(clf_names)}. Unable to create valid mapping for this case. Quiting...')
            return {}

        if self.verbose:
            logging.debug(f'Evaluating {len(clfs)} classifiers...')

        # first check if we actually have y data to evaluate on to prevent
        # trying to predict without data
        if self.y is None:
            logging.error('got no target data, can#t evaluate model score on \
target data')
            return {}

        scores = {}
        for i, clf in enumerate(clfs):
            classifier_name = clf_names[i]
            if self.verbose:
                logging.debug(f'currently evaluating {classifier_name}')
            clf_scores = self.cross_validate_model(clf,
                                                   self.x,
                                                   self.y)
            # print the scores if verbose
            if self.verbose:
                logging.debug(f'Mean overall score for {classifier_name} \
classifier was: {mean(clf_scores)*100} %')
                logging.debug(f'Max score for {classifier_name} classifier \
was: {max(clf_scores)*100} %')
                logging.debug(f'Min score for {classifier_name} classifier \
was: {min(clf_scores)*100} %')

            # add the classifier scores to the dict as well as the avg score
            mean_score = mean(clf_scores)
            scores[classifier_name] = tuple([mean_score, clf_scores])
        return scores

    def get_top_n_clfs(self,
                       clfs: dict[str, tuple[float, list[float]]],
                       n: int
                       ) -> dict[str, tuple[float, list[float]]]:
        """
        Selects from the map of classifiers and their respective scores
        the top n classifiers by mean performance. The number of elements
        it returns is hereby the minimum between the paramter n and the
        number of clfs it contains in the dictionary of all classifers.

        Paramters:
            - 'clfs' the hashmap that has to be created before from the
                evaluate model method containing all the classifiers and
                their average score as well as the list of scores
            - 'n' an integer defining how many of the top elements we want to
                return

        Return:
            - returns a striped version of the dictionary only containing the
                top n elements ordered by avg score.

        >>> e = Evaluator([1,2,3], [4,5,6], False)
        >>> e.get_top_n_clfs({'first': (0.5, [0.0, 1.0]),
        ...                    'second': (0.25, [0.25, 0.25]),
        ...                    'third': (0.0, [0.0, 0.0]),
        ...                    'fourth': (1.0, [1.0, 1.0]),
        ...                    'fifth': (0.75, [0.75, 0.75])},
        ...                     3) # doctest: +NORMALIZE_WHITESPACE
        {'fourth': (1.0, [1.0, 1.0]),
        'fifth': (0.75, [0.75, 0.75]),
        'first': (0.5, [0.0, 1.0])}
        >>> e.get_top_n_clfs({'first': (0.5, [0.0, 1.0]),
        ...                    'second': (0.25, [0.25, 0.25]),
        ...                    'third': (0.0, [0.0, 0.0]),
        ...                    'fourth': (1.0, [1.0, 1.0]),
        ...                    'fifth': (0.75, [0.75, 0.75])}, 0)
        {}
        >>> e.get_top_n_clfs({'first': (0.5, [0.0, 1.0]),
        ...                    'second': (0.25, [0.25, 0.25]),
        ...                    'third': (0.0, [0.0, 0.0]),
        ...                    'fourth': (1.0, [1.0, 1.0]),
        ...                    'fifth': (0.75, [0.75, 0.75])},
        ...                     6) # doctest: +NORMALIZE_WHITESPACE
        {'fourth': (1.0, [1.0, 1.0]),
        'fifth': (0.75, [0.75, 0.75]),
        'first': (0.5, [0.0, 1.0]),
        'second': (0.25, [0.25, 0.25]),
        'third': (0.0, [0.0, 0.0])}
        """

        num_returns = min(n, len(clfs))

        sorted_clfs = sorted(clfs.items(), key=lambda x: x[1][0], reverse=True)
        return dict(sorted_clfs[:num_returns])

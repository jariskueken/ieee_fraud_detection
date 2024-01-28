from typing import Callable
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from model.evaluator import Evaluator

import logging
import optuna


class Optimizer:
    """
    A simple class to run hyperparamter optimization on different model types
    using optuna
    """

    def __init__(
        self, train_data_X: np.ndarray,
        train_data_y: np.ndarray,
        dataset_description: str,
        verbose: bool = False
    ) -> None:
        # initalize the training data for later evaluation use
        self.train_data_X = train_data_X
        self.train_data_y = train_data_y
        self.dataset_description = dataset_description

        # initalize the evaluator
        self.evaluator = Evaluator(train_data_X, train_data_y, verbose)

    def get_objective_function(
        self, clf_identifier: str
    ) -> Callable[[optuna.trial.Trial, Evaluator, int], float]:
        """
        A simple function that returns for a given classifiert the correct
        objective function. Works for all types of classifiers if the
        objective function is defined in this class for this type otherwise
        returns none.
        """
        # diferentiate between sklearn clfs
        if clf_identifier == "RF":
            logging.debug(
                "got random forrest classifiert, optimization function exists"
            )
            objective = rf_optimizer

        return objective

    def run_hpo(self, clf_identifier: str,
                n_trials: int,
                cv_splits: int) -> None:
        """
        A base function to run hyperparamete optimization using optima for a
        given clf
        """

        # get the correct objective function for our clf
        objective = self.get_objective_function(clf_identifier)

        if objective is None:
            # return none if no objective function was provided
            logging.error(
                "failed to run hpo because no objective function was\
provided"
            )
            return

        # create a maximization study
        study = optuna.create_study(direction="maximize")
        logging.info(f'running trial in {n_trials} trials using \
{clf_identifier} classifier on {self.dataset_description} dataset')
        study.optimize(lambda trial: objective(trial,
                                               self.evaluator,
                                               cv_splits),
                       n_trials=n_trials)

        # just log out the best trial
        logging.info(
            f"Best trial for {clf_identifier} classifier was \
{study.best_trial}"
        )


"""
In the following section are definitions for all optimization functions
that can be used
"""


def rf_optimizer(trial: optuna.trial.Trial,
                 evaluator: Evaluator,
                 cv_splits: int) -> float:
    rf_max_depth = trial.suggest_int("max_depth", 3, 20)
    rf_estimators = trial.suggest_int("n_estimators", 500, 3000)
    rf_max_features = trial.suggest_int("max_features", 1, 7)
    rf_random_state = trial.suggest_int("random_state", 1, 1000)

    rf_clf = RandomForestClassifier(max_depth=rf_max_depth,
                                    n_estimators=rf_estimators,
                                    max_features=rf_max_features,
                                    random_state=rf_random_state)
    score = evaluator.hold_out_evaluate(rf_clf,
                                        'rf_objective',
                                        0.4)

    return score

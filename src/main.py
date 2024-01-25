import argparse
import coloredlogs
import logging
import sys

import numpy as np

from typing import Any

from dataengineering.data_preprocessing import DataPreprocessor
from models.evaluator import Evaluator, SKF, TSS
from models.predictor import Predictor
from models.model_builder import ModelBuilder
from models.optimizer import Optimizer

from consts.model_consts import CLFS_SHORT_DICT, CLASSIFIERS_DICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict",
        action="store_true",
        help="if set to true we want to make a prediction on the test set. \
            Needs to be set to true to start a prediction otherwise we wont \
                make one"
    )
    parser.add_argument(
        "--traindata",
        type=str,
        required=True,
        help="path to the csv-files that contain the training data for the \
        model"
    )
    parser.add_argument(
        "--testdata",
        type=str,
        default="",
        help="path to the csv-files that contain the test data for the model"
    )
    parser.add_argument(
        "-st",
        "--submissiontemplate",
        type=str,
        help="path to the submission file template for the current competition"
    )
    parser.add_argument(
        "-sd",
        "--submissiontarget",
        type=str,
        help="path to the directory where the submissions should be stored"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="if set to true a ensemble of each base model will be build \
        and evaluated afterwards"
    )
    parser.add_argument(
        "-n",
        "-top-n-models",
        type=int,
        default=5,
        help="defines how many of all evaluated models will be used onwards to\
            build ensemble and predict"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="if set to true the output of every step is very verbose and \
        give as much information as possible"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="if set to true we start in debug mode ignoring parts of the code"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="if set to true we run cross validation on all models to \
determine scores on all models"
    )
    parser.add_argument(
        "--dataset-description",
        type=str,
        help="description of the dataset currently in use"
    )
    parser.add_argument(
        "-md",
        "--modeltarget",
        type=str,
        required=True,
        help="The path to the directory where the models should be stored"
    )
    parser.add_argument(
        "-hpo",
        action="store_true",
        help="Boolean flag to define if we should run hyperparamter \
    optimization"
    )
    return parser.parse_args()


def evaluate(data_X: np.ndarray,
             data_y: np.ndarray,
             clfs: dict[str, Any],
             n: int,
             verbose: bool) -> None:
    # predict on the given estimators
    evaluator = Evaluator(data_X,
                          data_y,
                          verbose)

    logging.info('evaluating base models...')
    # evaluate the scores on the list of all classifiers
    clf_scores = evaluator.evaluate_model(list(clfs.values()),
                                          list(clfs.keys()),
                                          TSS,
                                          10)

    # Return the scores of the top five
    # parse the score into readable output
    top_n_clf_scores = evaluator.get_top_n_clfs(clf_scores,
                                                n)

    for score in top_n_clf_scores:
        logging.info(f'Average score of {score}-classifier is \
                {round(top_n_clf_scores[score][0], 3) * 100}%')


def optimize(data_X: np.ndarray,
             data_y: np.ndarray,
             dataset_description: str,
             clf_identifier: str,
             n_trials: int,
             verbose: bool):

    opt = Optimizer(data_X,
                    data_y,
                    dataset_description,
                    verbose)

    # runs hpo and loggs the result
    opt.run_hpo(clf_identifier,
                n_trials,
                3  # num cv splits
                )


# TODO: wrap main method better
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname) -10s %(asctime)s %(module)s:\
%(lineno)s %(funcName)s %(message)s",
                        level=logging.DEBUG)
    coloredlogs.install(level='DEBUG', fmt="%(levelname) -10s %(asctime)s \
%(module)s:%(lineno)s %(funcName)s %(message)s")
    preprocessor = DataPreprocessor(args.traindata,
                                    args.testdata,
                                    'isFraud',
                                    args.predict,
                                    args.verbose)
    # TODO: better feature selection method, select good features but only
    # later after we tested all models and found the one we want to work with

    logging.info('Preprocessing Data...')
    # preprocess the data
    preprocessor.preprocess_data()

    mb = ModelBuilder(
        preprocessor.train_data_X,
        preprocessor.train_data_y,
        args.modeltarget,
        args.verbose
    )

    # run hpo
    if args.hpo:
        optimize(preprocessor.train_data_X,
                 preprocessor.train_data_y,
                 args.dataset_description,
                 "RF",
                 100,
                 args.verbose)

    # only evaluate cv scores if we want to
    if args.evaluate:
        evaluate(preprocessor.train_data_X,
                 preprocessor.train_data_y,
                 CLASSIFIERS_DICT,
                 args.n,
                 args.verbose)

    # HACK: currently use all clfs, needs to be fixed to predict only on
    # subset of clfs
    if args.predict:
        for clf_name in CLASSIFIERS_DICT:
            # bool to define if we are currently testint an ensemble modle to
            # prevent retraining the model in this case
            # is_ensemble = False
            clf = CLASSIFIERS_DICT[clf_name]
            # just use the clf name as identifier with underscores replacing
            # spaces
            clf_identifier = str(clf).replace(' ', '_')

            # get the classifier from the model builder
            clf, pretrained = mb.get_classifier(clf,
                                                clf_identifier,
                                                args.dataset_description)

            # make a prediction for the top 5 clfs
            if args.predict:
                if preprocessor.train_data_y is None:
                    logging.error('failed to predict with empty training \
target')
                    break

                predictor = Predictor(preprocessor.train_data_X,
                                      preprocessor.train_data_y,
                                      preprocessor.test_data_X,
                                      args.dataset_description,
                                      mb,
                                      args.verbose)

                logging.info(f"predicting on {clf_identifier}")
                pred_path, prediction = predictor.predict(
                    clf,  # here we need to use the
                    # identifier from the dict
                    clf_identifier,
                    False,
                    True,
                    args.submissiontemplate,
                    args.submissiontarget,
                    'isFraud',
                    pretrained
                )

    # if we have set ensemble to true we want to create an ensemble of the top n clfs
    if args.ensemble:
        # get all clfs in a lst
        clfs = CLASSIFIERS_DICT
        logging.info(f'building ensemble of top {args.n} classifiers')
        mb = ModelBuilder(preprocessor.train_data_X,
                          preprocessor.train_data_y,
                          args.modeltarget,
                          args.verbose)

        # stacking classifier
        sclf = mb.build_stacking_ensemble(list(clfs.values()),
                                          list(clfs.keys()))
        sclf_identifier = f'stackin_ensemble-\
{"_".join([name for name in list(clfs.keys())])}'

        # stacking classifier using all features
        sclf_af = mb.build_stacking_ensemble(list(clfs.values()),
                                             list(clfs.keys()),
                                             True)
        sclf_af_identifier = f'stackin_ensemble_af-\
{"_".join([name for name in list(clfs.keys())])}'

        # voting classifier
        eclf = mb.build_voting_ensemble(list(clfs.values()),
                                        list(clfs.keys()))
        eclf_identifier = f'voting_ensemble-\
{"_".join([name for name in list(clfs.keys())])}'

        ensembles = {eclf_identifier: eclf,
                     sclf_identifier: sclf,
                     sclf_af_identifier: sclf_af}

        # evaluate the ensemble models
        evaluate(preprocessor.train_data_X,
                 preprocessor.train_data_y,
                 ensembles,
                 args.n,
                 True)

        predictor = Predictor(preprocessor.train_data_X,
                              preprocessor.train_data_y,
                              preprocessor.test_data_X,
                              args.dataset_description,
                              args.verbose)

        for eclf_identifier in ensembles:
            logging.info(f"predicting on {eclf_identifier}")
            pred_path, prediction = predictor.predict(
                eclf,  # here we need to use the
                # identifier from the dict
                ensembles[eclf_identifier],
                False,
                True,
                args.submissiontemplate,
                args.submissiontarget,
                'isFraud',
                False,
            )


if __name__ == "__main__":
    main(parse_args())

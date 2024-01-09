import argparse
import coloredlogs
import logging
import sys

from dataengineering.data_preprocessing import DataPreprocessor
from models.evaluator import Evaluator
from models.predictor import Predictor
from models.model_builder import ModelBuilder

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
        "-md",
        "--modeltarget",
        type=str,
        help="The path to the directory where the models should be stored"
    )
    return parser.parse_args()


def evaluate(data,
             n: int,
             verbose: bool) -> None:
    # predict on the given estimators
    evaluator = Evaluator(data[0],
                          data[1],
                          verbose)

    logging.info('evaluating base models...')
    # evaluate the scores on the list of all classifiers
    clf_scores = evaluator.evaluate_model(list(CLASSIFIERS_DICT.values()),
                                          list(CLASSIFIERS_DICT.keys()))

    # Return the scores of the top five
    # parse the score into readable output
    top_n_clf_scores = evaluator.get_top_n_clfs(clf_scores,
                                                n)

    for score in top_n_clf_scores:
        logging.info(f'Average score of {score}-classifier is \
                {round(top_n_clf_scores[score][0], 3) * 100}%')


# TODO: wrap main method better
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname) -10s %(asctime)s %(module)s:\
%(lineno)s %(funcName)s %(message)s",
                        level=logging.DEBUG)
    coloredlogs.install(level='DEBUG', fmt="%(levelname) -10s %(asctime)s \
%(module)s:%(lineno)s %(funcName)s %(message)s")
    preprocessor = DataPreprocessor(args.verbose)
    # TODO: better feature selection method, select good features but only
    # later after we tested all models and found the one we want to work with

    logging.info('Preprocessing Data...')
    data = preprocessor.preprocess_data(args.traindata, 'isFraud')

    mb = ModelBuilder(
        data[0],
        data[1],
        args.modeltarget,
        args.verbose
    )

    # only evaluate cv scores if we want to
    if args.evaluate:
        evaluate(data,
                 args.n,
                 args.verbose)

    # HACK: currently use all clfs, needs to be fixed to predict only on
    # subset of clfs
    test_data = preprocessor.preprocess_data(args.testdata)
    for clf_name in CLASSIFIERS_DICT:
        # bool to define if we are currently testint an ensemble modle to
        # prevent retraining the model in this case
        # is_ensemble = False
        clf = CLASSIFIERS_DICT[clf_name]
        # just use the clf name as identifier with underscores replacing spaces
        clf_identifier = str(clf).replace(' ', '_')

        # get the classifier from the model builder
        clf, pretrained = mb.get_classifier(clf,
                                            clf_identifier,
                                            'baseset')

        # make a prediction for the top 5 clfs
        if args.predict:
            if data[1] is None:
                logging.error('failed to predict with empty training target')
                break

            predictor = Predictor(data[0],
                                  data[1],
                                  test_data[0],
                                  'baseset',
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

    """
    # if we have set ensemble to true we want to create an ensemble of the top n clfs
    if args.ensemble:
        # get all clfs in a lst
        clfs = {}
        for score in top_n_clf_scores:
            clfs[score] = CLASSIFIERS_DICT[score]
        logging.info(f'building ensemble of top {args.n} classifiers')
        mb = ModelBuilder(data[0],
                          data[1],
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

        ensembles = [sclf, sclf_af, eclf]
        ensembles_identifiers = [eclf_identifier,
                                 sclf_identifier,
                                 sclf_af_identifier]
        # evaluate the ensemble models
        scores = evaluator.evaluate_model(ensembles,
                                          ensembles_identifiers)
        for score in scores:
            logging.info(f'median score of {score} classifier is \
                {round(scores[score][0], 3) * 100}%')

        if args.predict:
            test_data = preprocessor.preprocess_data(args.testdata)
            predictor = Predictor(data[0], data[1], test_data[0], args.verbose)

            for eclf, eclf_identifier in zip(ensembles, ensembles_identifiers):
                logging.info(f"predicting on {eclf_identifier}")
                pred_path, prediction = predictor.predict(
                    eclf,  # here we need to use the
                    # identifier from the dict
                    eclf_identifier,
                    False,
                    True,
                    args.submissiontemplate,
                    args.submissiontarget,
                    'isFraud',
                    False,
                )
    """


if __name__ == "__main__":
    main(parse_args())

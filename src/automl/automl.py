from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from datetime import datetime
import os


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
        required=True,
        help="path to the csv-files that contain the test data for the model"
    )
    parser.add_argument(
        "-st",
        "--submissiontemplate",
        type=str,
        required=True,
        help="path to the submission file template for the current competition"
    )
    parser.add_argument(
        "-sd",
        "--submissiontarget",
        type=str,
        required=True,
        help="path to the directory where the submissions should be stored"
    )
    parser.add_argument(
        "--datasetidentifier",
        type=str,
        help="description of the dataset currently in use"
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=1200.0,
        help="The maximum time for autogluon to train on the trainingset"
    )
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    print(args.__dict__)
    # read the csv data into tabular dataset
    train_data = TabularDataset(args.traindata)
    
    timelimit = args.timelimit

    # create a validation split
    train, validation = train_test_split(train_data, test_size=0.3)
    
    # fit
    predictor = TabularPredictor(label='isFraud',
                                 problem_type='binary',
                                 eval_metric='roc_auc',
                                 verbosity=4
                                 ).fit(train,
                                       validation,
                                       use_bag_holdout=True,
                                       presets='best_quality',
                                       time_limit=timelimit, # 20 Minutes per model default, TODO: increase later to 7200s / 120 min
                                 )

    # delete the train data from memory and read the test data
    del train_data
    test_data = TabularDataset(args.testdata)

    # predict proba
    y_pred = predictor.predict_proba(test_data)  # returns a dataframe of cols 0, 1
    del test_data
    pd.DataFrame(y_pred).to_csv('submission.csv')
    submission = pd.read_csv(args.submissiontemplate, index_col=None)
    # overwrite the target in the predicitions
    submission['isFraud'] = y_pred[1]  # just write the true prediction column to the submission
    
    # create suibmission name
    # Create the submissions for each day folder if it doesn't exist
    date = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join(args.submissiontarget, date)):
        os.makedirs(os.path.join(args.submissiontarget, date))

    # generate the output file name, format the date aftetr ISO8601
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    output_file = f'submission_{timestamp}_autogluon{int(timelimit)}_\
_bestqualitypreset_{args.datasetidentifier}.csv'
    output_path = os.path.join(args.submissiontarget, date, output_file)
    
    # write the result
    submission.to_csv(output_path, index=False)


if __name__ == "__main__":
    main(parse_args())
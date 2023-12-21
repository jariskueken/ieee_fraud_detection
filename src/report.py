import logging
import sys
import os
import argparse

import pandas as pd
from ydata_profiling import ProfileReport
from pymfe.mfe import MFE

from util import file_util


class Report:
    """
    A simple class to generate profiling reports about a given dataset
    """

    def __init__(self,
                 path: str,
                 output_path: str) -> None:
        """
        Takes as input a path which is then checked for correctness if it
        exists and if it is a path to csv file. Also creates an output path
        to where the output files should be stored.

        Properties:
            - 'self.path' the path to the data file
            - 'self.output_path' the path to the output file directory
            - 'self.data' the data as pandas dataframe
            - 'self.name' the basename of the data file
            - 'self.feature_columns' the list of column names for the
                dataframe that contain the features
            - 'self.target_colum' the name of the target column for the
                dataframe
        """

        self.path = path
        self.output_path = output_path
        error = file_util.check_file_path_exists(self.path)
        if error is not None:
            logging.error(f'Received in error trying to open the data file: \
{error}')
            return
        error = file_util.check_file_is_csv(self.path)
        if error is not None:
            logging.error(f'received a non csv file as data input file\
aborting... {error}')
            return

        # create the output path if it does not exists yet
        if not os.path.exists(self.output_path):
            logging.debug(f'output directory does not exists, creating now at \
{self.output_path}...')
            os.makedirs(self.output_path)

        data = pd.read_csv(self.path)
        self.data = data
        self.name = os.path.basename(self.path)

        # features and targets
        self.feature_columns = []
        self.target_colum = ''

    def pandas_profiling_report(self) -> str:
        """
        Creates a pandas profiling report for a given dataset

        Return:
            - a path to the created html report
        """

        profile = ProfileReport(self.data, title='Pandas Profiling Report')
        logging.debug(f'Creating profile report html for: {self.name}')
        output_path = os.path.join(self.output_path,
                                   f'{self.name}_pandas_report.html')
        profile.to_file(output_path)
        logging.debug(f'Successfully stored report at {output_path}')
        return output_path

    def mfe_report(self) -> str:
        """
        Creates a report about the dataset using the python meta feature
        extractor package
        """
        mfe = MFE()

        # fit the data to the mfe, requires nparrays as datatypes
        mfe.fit(self.data[self.feature_columns].to_numpy(),
                self.data[self.target_colum].to_numpy())
        logging.debug(f'fitted data and extracting meta features for \
{self.name}')

        # extract the features and zip into a dict where the key is the metric
        # and the value is the corresponding value
        features = mfe.extract()

        # turn the results into a dictionary
        stats = dict(zip(features[0], features[1]))

        output_path = os.path.join(self.output_path,
                                   f'{self.name}_mfe_stats.txt')

        with open(output_path, "a") as f:
            for key in stats:
                f.write(f"Key: {key}, Value: {stats[key]} \n")
        logging.debug(f'Successfully stored report at {output_path}')

        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to the csv-files that contain the data that has to be\
analized"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="patht to output directory where the data should be stored"
    )
    parser.add_argument(
        "--profiling",
        action='store_true',
        help="creates the pandas profiling report"
    )
    parser.add_argument(
        "--mfe"
        ""
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname) -10s %(asctime)s %(module)s:\
%(lineno)s %(funcName)s %(message)s",
                        level=logging.DEBUG)

    r = Report(args.data,
               args.output_dir)

    if args.profiling:
        output_path = r.pandas_profiling_report()
        logging.info(f'created pandas profiling report for \
{os.path.basename(args.data)} at {output_path}')

    if args.mfe:
        output_path = r.mfe_report()
        logging.info(f'created python meta feature extractor report for \
{os.path.basename(args.data)} at {output_path}')


if __name__ == '__main__':
    main(parse_args())

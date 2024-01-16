import pandas as pd
import logging
import os
import sys
import argparse

from sklearn.preprocessing import LabelEncoder

STRATEGY_MEAN = 'mean'
STRATEGY_MEDIAN = 'medain'
STRATEGY_RANDOM = 'random'
STRATEGY_999 = 'strategy_999'  # fill missing values with -999


class DataPreperator:
    """
    A simple class to preperate a given dataset such that it can be used in a
    ml base model
    """

    def __init__(self,
                 paths: list[str],
                 match: str,
                 write: bool = False,
                 replace_strategy: str = STRATEGY_999,
                 output_path: str = '') -> None:
        """
        Intitilizes the class. Takes as input a list of paths to the data
        files and a string containing the name of the column that the
        dataframes should be matched on.
        """
        self.paths = paths
        self.match = match
        self.dataframes: list[pd.DataFrame]
        self.data: pd.DataFrame
        self.output_path = output_path
        self.replace_strategy = replace_strategy

        # read the data into memory
        self.__read_data()

        # merge the dataframes into one unified dataframe
        self.__merge_data()

        # TODO: fix so that we dont automatically run all steps on creation
        # strip down the data
        # self.remove_non_numerical_columns()
        self.remove_empty_columns()

        self.replace_empty_values(self.replace_strategy)

        self.encode_categorical_features()

        # write the data into a new csv file
        if write:
            if self.output_path == '':
                raise ValueError('Cannot write to empty output path')
            else:
                self.__write_to_csv()

    def __read_data(self) -> None:
        """
        Reads the data from the input paths. Assumes that the data is in csv
        format.
        """
        logging.debug('reading datafiles into dataframes')
        self.dataframes = [pd.read_csv(path, index_col=None) for path in self.paths]

    def __merge_data(self) -> None:
        """
        Merges the dataframes in the class into one dataframe.
        """
        logging.debug(f'merging all dataframe into one dataframe on\
{self.match}')
        self.data = self.dataframes[0]
        for df in self.dataframes[1:]:
            if self.match in df.columns:
                # use outer as merge type to perserv columns from both dfs
                self.data = self.data.merge(df, on=self.match, how='left')
            else:
                raise ValueError(f'Could not find the match column\
{self.match}')

    def remove_empty_columns(self,
                             threshold: float = 0.9) -> None:
        """
        removes all columns from the dataframe that contain more then a
        threshold value of NaN. So we drop columns that are mostly empty
        """

        # store the shape to later check the threshold. shape is tuple of
        # [numRows, numCols]
        shape = self.data.shape
        for col in self.data:
            empty = self.data[col].isna().sum() / shape[0]
            # drop the column if the threshold is exceeded
            if empty >= threshold:
                logging.info(f'column {col} contains {round(empty * 100, 6)}% NaN.\
Dropping...')
                del self.data[col]

    def remove_non_numerical_columns(self) -> None:
        """
        removes all non numerical columns from the dataframe
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        for col in self.data:
            # drop if a column is not of one of the predefined types
            if self.data[col].dtype not in numerics:
                logging.debug(f'column {col} is of type {self.data[col].dtype}\
, dropping...')
                del self.data[col]

    def encode_categorical_features(self) -> None:
        """
        encodes every categroical feature in the dataset using the sklearn
        label encoder
        """
        le = LabelEncoder()
        for col in self.data:
            if self.data[col].dtype == 'object':
                encoded_col = le.fit_transform(self.data[col])
                self.data[col] = encoded_col

    def replace_empty_values(self,
                             strategy: str = STRATEGY_MEAN) -> None:
        """
        replaces all NaN values of a column with a statistical property of the
        respective column. Handles the following replacement strategies:
            - mean: uses the mean of the column and replaces every missing
                value with the mean (good of normal distribution)
            - median: uses the median of the column and replaces every missing
                values with the median (may be better if we don't have
                normal distribution)
            - random samples:
            TODO: implement selects random samples from the
                already existing data of that column
                ref: https://www.datacamp.com/tutorial/techniques-to-handle-missing-data-values
        """
        for col in self.data:
            # only select columns that contain missing values
            if self.data[col].isna().any():
                # define the value that will fill the missing values
                filler = 0
                if strategy == STRATEGY_MEAN:
                    filler = self.data[col].mean()
                elif strategy == STRATEGY_MEDIAN:
                    filler = self.data[col].median()
                elif strategy == STRATEGY_RANDOM:
                    # TODO: implement
                    raise NotImplementedError
                elif strategy == STRATEGY_999:
                    # if we have categorical features fill with missing
                    if self.data[col].dtype == 'object':
                        filler = 'missing'
                    else:
                        filler = -999
                logging.debug(f'filling column {col} of dtype \
{self.data[col].dtype} with {strategy} and value {filler} for \
{self.data[col].isna().sum()} entries')
                # fill the missing values
                self.data[col] = self.data[col].fillna(filler)

    def __write_to_csv(self) -> None:
        # TODO: collect the correct name for the filepath
        # FIXME
        data_type = ''
        if 'train' in self.paths[0]:
            data_type = 'train'
        elif 'test' in self.paths[0]:
            data_type = 'test'
        filepath = f'{data_type}_data_prepared.csv'
        # store the cleaned data into a csv file
        self.data.to_csv(os.path.join(self.output_path, filepath), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        nargs="+",
        default=[],
        required=True,
        help="paths to the csv-files that contain the data that has to be\
prepared"
    )
    parser.add_argument(
        "--match",
        type=str,
        required=True,
        help="the name of the column on which the different data files should\
be matched"
    )
    parser.add_argument(
        "-w",
        action="store_true",
        default=False,
        help="defines if the data should be written and stored into a csv\
file, should only be used if output_dir is also present"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default='',
        help="path to output directory where the data should be stored"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(stream=sys.stdout,
                        format="%(levelname) -10s %(asctime)s %(module)s:\
%(lineno)s %(funcName)s %(message)s",
                        level=logging.DEBUG)

    _ = DataPreperator(args.data,
                       args.match,
                       args.w,
                       STRATEGY_999,
                       args.output_dir)


if __name__ == '__main__':
    main(parse_args())

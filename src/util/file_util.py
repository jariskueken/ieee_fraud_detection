import re


class InvalidFileTypeError(Exception):
    pass


def check_file_path_exists(f: str) -> None | OSError:
    try:
        with open(f, 'r') as _:  # throws an OSError if file does not exist
            return None
    except OSError:
        return OSError  # type: ignore


def check_file_is_csv(f: str) -> None | InvalidFileTypeError:
    # check if the file is a csv file
    csv_path_regex = r'\.csv$'
    if re.search(csv_path_regex, f, re.IGNORECASE):
        return None
    else:
        return InvalidFileTypeError('Received file path does not \
                                    point to a csv file')


def check_file_is_pickle(f: str) -> None | InvalidFileTypeError:
    # check if the file is a csv file
    pkl_path_regex = r'\.pkl$'
    if re.search(pkl_path_regex, f, re.IGNORECASE):
        return None
    else:
        return InvalidFileTypeError('Received file path does not \
                                    point to a pkl file')

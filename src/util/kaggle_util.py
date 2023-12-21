import kaggle



def push_to_kaggle(filepath: str,
                    ):
    kaggle.api.competitions_submissions_upload(
        filepath
    )


if __name__ == "__main__":
    push_to_kaggle()
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import joblib


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load the data
    data = pd.read_csv(input_filepath)

    # Load the pipelines
    preprocessing = joblib.load("./src/features/preprocessing.joblib")

    # Apply the transformations
    X_transformed = pd.DataFrame(
        preprocessing.transform(data), columns=preprocessing.get_feature_names_out()
    )

    # Save the transformed data
    with open(output_filepath, "wb") as f:
        joblib.dump(X_transformed, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
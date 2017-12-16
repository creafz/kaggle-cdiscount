from itertools import groupby
import os

import click
from click import option as opt
import numpy as np
from scipy.stats import gmean
import pandas as pd
from tqdm import tqdm

import config
from prepare_data import get_category_mapping
from utils import load_pickle, npz_directory_iterator


@click.command()
@opt('--directory', type=str, required=True)
@opt('--averaging', type=click.Choice(['mean', 'gmean']), default='mean')
@opt('--submission-filename', type=str, required=True)
def main(
    directory,
    averaging,
    submission_filename,
):
    mean = np.mean if averaging == 'mean' else gmean
    label_to_cat_id, cat_id_to_label = get_category_mapping()
    known_categories_for_test_products = load_pickle(
        config.KNOWN_CATEGORIES_FOR_TEST_PRODUCTS_FILENAME
    )
    test_data = load_pickle(config.TEST_DATA_FILENAME)
    predictions_keys = [
        (product_id, img_num) for *_, product_id, img_num in test_data
    ]
    zipper = zip(
        predictions_keys,
        npz_directory_iterator(
            os.path.join(config.PREDICTIONS_PATH, directory),
        ),
    )
    grouper = groupby(zipper, key=lambda row: row[0][0])
    categories = {}
    for group, items in tqdm(grouper, total=1018721):
        probs = mean(np.array([item[1] for item in items]), axis=0)
        label = np.argmax(probs)
        category_id = label_to_cat_id[label]
        categories[group] = category_id

    known_categories = {
        product_id: label_to_cat_id[label] for product_id, label
        in known_categories_for_test_products.items()
    }
    categories.update(known_categories)
    assert len(categories) == 1768182

    df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    df.drop('category_id', inplace=True, axis=1)
    predictions_df = pd.DataFrame(
        list(categories.items()),
        columns=['_id', 'category_id'],
    )
    predictions_df['_id'] = predictions_df['_id'].astype(int)
    df = df.merge(predictions_df, on=['_id'])
    df.to_csv(
        os.path.join(config.SUBMISSIONS_PATH, submission_filename),
        index=False,
    )


if __name__ == '__main__':
    main()

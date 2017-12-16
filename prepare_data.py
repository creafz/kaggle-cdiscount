from itertools import groupby
import os
import struct

import bson
import click
from click import option as opt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tqdm import tqdm

import config
import utils


def make_category_mapping():
    print('Processing data for category mapping')
    prod_to_category = {}
    with open(config.TRAIN_BSON_FILE, 'rb') as bson_file:
        bson_data = bson.decode_file_iter(bson_file)
        for product in tqdm(bson_data, total=7069896):
            product_id = product['_id']
            category_id = product['category_id']
            prod_to_category[product_id] = category_id
    prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
    prod_to_category.index.name = '_id'
    prod_to_category.rename(columns={0: 'category_id'}, inplace=True)
    prod_to_category.to_csv(config.PROD_TO_CATEGORY_CSV_PATH)


def get_category_mapping():
    df = pd.read_csv(config.PROD_TO_CATEGORY_CSV_PATH)
    id_to_category_mapping = dict(
        enumerate(
            sorted(
                set(df['category_id'].values)
            )
        )
    )
    category_to_id_mapping = {
        category: id_ for id_, category in id_to_category_mapping.items()
    }
    return id_to_category_mapping, category_to_id_mapping


def get_product_category_dict():
    _, category_to_id_mapping = get_category_mapping()
    prod_to_category = pd.read_csv(config.PROD_TO_CATEGORY_CSV_PATH)
    prod_to_category['category_id'] = prod_to_category['category_id'].map(
        category_to_id_mapping,
    )
    product_category_dict = dict(
        zip(
            prod_to_category['_id'],
            prod_to_category['category_id'],
        ),
    )
    return product_category_dict


def read_bson(bson_path, num_records):
    """
    Based on https://www.kaggle.com/inversion/processing-bson-files
    """
    data = []
    with open(bson_path, 'rb') as f, tqdm(total=num_records) as pbar:
        pbar.set_description(
            'Extracting data from ' + os.path.basename(bson_path)
        )
        offset = 0
        records_read = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack('<i', item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item['_id']
            num_imgs = len(item['imgs'])

            for img_num in range(num_imgs):
                sha1_hash = utils.get_sha1_hash(
                    item['imgs'][img_num]['picture']
                )
                data.append(
                    (offset, length, product_id, img_num, sha1_hash),
                )
            offset += length
            f.seek(offset)
            records_read += 1
            pbar.update()
    pbar.close()
    return data


@click.command()
@opt('--shuffle-seed', default=config.SHUFFLE_SEED)
@opt('--validation-data-ratio', default=0.05)
def main(
     shuffle_seed,
     validation_data_ratio,
):
    make_category_mapping()
    train_bson_data = read_bson(config.TRAIN_BSON_FILE, num_records=7069896)
    test_bson_data = read_bson(config.TEST_BSON_FILE, num_records=1768182)

    train_bson_data_images_count = len(train_bson_data)
    unique_train_bson_data_images_count = len(
        {sha1_hash for *_, sha1_hash in train_bson_data}
    )

    # There are 12371293 images in train.bson,
    # but only 7772910 of them are unique.
    print(
        f'There are {train_bson_data_images_count} images in train.bson, '
        f'but only {unique_train_bson_data_images_count} of them are unique.'
    )

    product_id_category = get_product_category_dict()

    train_data = []
    seen_hashes = set()
    hash_to_category = {}
    stream = tqdm(train_bson_data, desc='Processing data from train.bson')
    for offset, length, product_id, img_num, sha1_hash in stream:
        if sha1_hash in seen_hashes:
            continue
        seen_hashes.add(sha1_hash)
        category = product_id_category[product_id]
        hash_to_category[sha1_hash] = category
        train_data.append(
            ('train.bson', offset, length, product_id, img_num, category)
        )

    test_data = []
    known_categories_for_test_products = {}

    # Group images by product_id
    for product_id, imgs in tqdm(
        groupby(test_bson_data, key=lambda img: img[2]),
        total=1768182,
        desc='Processing data from test.bson',
    ):
        imgs = list(imgs)
        known_categories = [
            (sha1_hash, hash_to_category.get(sha1_hash))
            for *_, sha1_hash in imgs
            if hash_to_category.get(sha1_hash)
        ]
        if not known_categories:
            for *img_info, sha1_hash in imgs:
                test_data.append(tuple(img_info))
        else:
            known_category = known_categories[0]
            _, category = known_category
            known_categories_for_test_products[product_id] = category
            known_hashes = {
                sha1_hash for sha1_hash, category in known_categories

            }
            imgs_info = [
                img_info for *img_info, sha1_hash in imgs
                if sha1_hash not in known_hashes
            ]
            for img_info in imgs_info:
                train_data.append(('test.bson', *img_info, category))

    total_train_images_count = len(train_data)
    train_images_from_train_bson_count = len(
        [filename for filename, *_ in train_data if filename == 'train.bson']
    )
    train_images_from_test_bson_count = (
        total_train_images_count - train_images_from_train_bson_count
    )

    # There are 7930253 images in train_data, 7772910 from train.bson
    # and 157343 from test.bson
    print(
        f'There are {total_train_images_count} images in train_data, '
        f'{train_images_from_train_bson_count} from train.bson '
        f'and {train_images_from_test_bson_count} from test.bson'
    )

    # Count number of products in test.bson
    test_bson_products_count = len({img[2] for img in test_bson_data})
    test_products_with_unknown_categories_count = (
        test_bson_products_count - len(known_categories_for_test_products)
    )

    # There are 1768182 products in test.bson, and we know categories
    # for 749461 of them. So we need to predict categories
    # for 1018721 products from 1625110 images.
    print(
        f'There are {test_bson_products_count} products in test.bson, and '
        f'we know categories for {len(known_categories_for_test_products)} '
        f'of them. So we need to predict categories '
        f'for {test_products_with_unknown_categories_count} products '
        f'from {len(test_data)} images.'
    )

    shuffled_train_data = shuffle(train_data, random_state=shuffle_seed)

    n_splits = int(1 / validation_data_ratio)

    # Splitting shuffled_train_data into 20 folds. Using the first fold for
    # validation and all the rest folds for training.
    print(
        f'Splitting shuffled_train_data into {n_splits} folds. '
        f'Using the first fold for validation and all the rest '
        f'folds for training.'
    )
    skf = StratifiedKFold(n_splits=n_splits)
    split = skf.split(
        shuffled_train_data,
        [category for *_, category in shuffled_train_data],
    )
    train_index, valid_index = next(split)
    utils.save_pickle(
        {
            'shuffled_train_data': shuffled_train_data,
            'train_index': train_index,
            'valid_index': valid_index,
        },
        config.TRAIN_VALID_DATA_FILENAME,
        verbose=True,
    )
    utils.save_pickle(test_data, config.TEST_DATA_FILENAME, verbose=True)
    utils.save_pickle(
        known_categories_for_test_products,
        config.KNOWN_CATEGORIES_FOR_TEST_PRODUCTS_FILENAME,
        verbose=True,
    )


if __name__ == '__main__':
    main()

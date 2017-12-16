import bson
import cv2
import numpy as np
from torch.utils.data import Dataset

import config
import utils


class CdiscountDataset(Dataset):
    """
    Based on https://www.kaggle.com/alekseit/pytorch-bson-dataset
    """

    @staticmethod
    def read_bson_data(filepath, offset, length):
        with open(filepath, 'rb') as f:
            f.seek(offset)
            return bson.BSON.decode(f.read(length))

    @staticmethod
    def decode_img(bson_data, img_num):
        byte_str = bson_data['imgs'][img_num]['picture']
        return cv2.imdecode(
            np.fromstring(byte_str, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )


class CdiscountTrainDataset(CdiscountDataset):

    def __init__(self, bson_filepaths, transform=None, mode='train'):
        assert mode in {'train', 'valid'}
        self.transform = transform
        train_valid_data = utils.load_pickle(config.TRAIN_VALID_DATA_FILENAME)
        self.bson_filepaths = bson_filepaths
        self.dataset_index = train_valid_data[f'{mode}_index']
        self.data = train_valid_data['shuffled_train_data']

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, i):
        index = self.dataset_index[i]
        filename, offset, length, product_id, img_num, category = self.data[index]
        filepath = self.bson_filepaths[filename]
        bson_data = self.read_bson_data(filepath, offset, length)
        img = self.decode_img(bson_data, img_num)
        if self.transform is not None:
            img = self.transform(img)
        return img, category


class CdiscountTestDataset(CdiscountDataset):

    def __init__(self, bson_filepath, transform=None):
        self.transform = transform
        self.bson_filepath = bson_filepath
        self.data = utils.load_pickle(config.TEST_DATA_FILENAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        offset, length, product_id, img_num = self.data[i]
        bson_data = self.read_bson_data(self.bson_filepath, offset, length)
        img = self.decode_img(bson_data, img_num)
        if self.transform is not None:
            img = self.transform(img)
        return img, product_id, img_num

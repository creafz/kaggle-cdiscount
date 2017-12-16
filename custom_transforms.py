import itertools

import cv2
from imgaug import augmenters as iaa
import numpy as np


class CustomTransform:

    def _pre_call_hook(self):
        pass

    def __call__(self, image):
        self._pre_call_hook()
        return self.transform(image)

    def transform(self, image):
        raise NotImplementedError()


class Resize(CustomTransform):

    def __init__(self, output_size):
        self.output_size = output_size

    def transform(self, image):
        return cv2.resize(image, self.output_size)


class Lambda(CustomTransform):

    def __init__(self, func):
        self.func = func

    def transform(self, image):
        return self.func(image)


class ExpandDims(CustomTransform):

    def __init__(self, axis):
        self.axis = axis

    def transform(self, image):
        return np.expand_dims(image, self.axis)


class ImgAug(CustomTransform):

    def __init__(self, augmenters, p=None):
        if not isinstance(augmenters, list):
            augmenters = [augmenters]
        seq = iaa.Sequential(augmenters)
        if p is not None:
            seq = iaa.Sometimes(p, seq)
        self.seq = seq

    def transform(self, image):
        return self.seq.augment_image(image)


class FlipRotate(CustomTransform):

    def __init__(self):
        flip_options = (False, True)
        rotate_options = range(4)
        self.options = list(itertools.product(flip_options, rotate_options))

    def _pre_call_hook(self):
        option_index = np.random.randint(0, len(self.options))
        self.option = self.options[option_index]

    def transform(self, image):
        flip, rotate_k = self.option
        if flip:
            image = np.fliplr(image)
        image = np.rot90(image, rotate_k)
        return image


class RandomCrop(CustomTransform):

    def __init__(self, original_size, crop_size):
        self.original_size = original_size
        self.crop_size = crop_size

    def _pre_call_hook(self):
        original_width, original_height = self.original_size
        width, height = self.crop_size
        x = np.random.randint(0, original_width - width - 1)
        y = np.random.randint(0, original_height - height - 1)
        self.crop_coords = np.s_[y: y + height, x: x + width]

    def transform(self, image):
        return image[self.crop_coords]


class Clip(CustomTransform):

    def __init__(self, min_value=0, max_value=255):
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, image):
        return np.clip(image, self.min_value, self.max_value)


class AsContiguousArray(CustomTransform):

    def transform(self, image):
        return np.ascontiguousarray(image)


class Crop(CustomTransform):

    def __init__(self, original_size, crop_size, crop_location):
        self.original_size = original_size
        self.crop_size = crop_size
        original_width, original_height = self.original_size
        width, height = self.crop_size
        pad_width = (original_width - width) // 2
        pad_height = (original_height - height) // 2
        crop_options = {
            'center': np.s_[
              pad_height: original_height - pad_height,
              pad_width: original_width - pad_width,
            ],
            'top_left': np.s_[:height, :width, :],
            'top_right': np.s_[:height, -width:, :],
            'bottom_left': np.s_[-height:, :width, :],
            'bottom_right': np.s_[-height:, -width:, :],
        }
        self.crop_coords = crop_options[crop_location]

    def transform(self, image):
        return image[self.crop_coords]


class CenterCrop(Crop):

    def __init__(self, original_size, crop_size):
        super().__init__(original_size, crop_size, crop_location='center')


class BGR2RGB(CustomTransform):

    def transform(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FlipLr(CustomTransform):

    def transform(self, image):
        return np.fliplr(image)


class FlipUd(CustomTransform):

    def transform(self, image):
        return np.flipud(image)

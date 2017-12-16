from imgaug import augmenters as iaa
from torchvision.transforms import Compose, Normalize, ToTensor

import config
import custom_transforms as t


train_augmentations = (
    t.ImgAug(iaa.Fliplr(0.5)),
    t.ImgAug(iaa.Flipud(0.5)),
    t.ImgAug(
        iaa.Affine(
            scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
            rotate=(-24, 24),
            shear=(-24, 24),
            order=[0, 1],
            mode='reflect'
        ),
        p=0.5,
    ),
    t.ImgAug(
        iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace='RGB', to_colorspace='HSV'),
            iaa.WithChannels([0], iaa.Add((-30, 30))),
            iaa.ChangeColorspace(from_colorspace='HSV', to_colorspace='RGB'),
        ]),
        p=0.25,
    ),
    t.Clip(),
)


def make_augmentation_transforms(img_size, mode):
    transforms = [
        t.BGR2RGB(),
    ]

    if img_size != config.ORIGINAL_IMG_SIZE:
        img_crop_transform = t.RandomCrop if mode == 'train' else t.CenterCrop
        transforms.append(
            img_crop_transform(
                config.ORIGINAL_IMG_SIZE,
                img_size,
            )
        )

    if mode == 'train':
        transforms += train_augmentations

    transforms += [
        ToTensor(),
        Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ]

    return Compose(transforms)


def make_test_augmentation_transforms(
    crop_params=None,
    flip_lr=False,
    flip_ud=False,
):
    transforms = [
        t.BGR2RGB(),
    ]

    if crop_params is not None:
        crop_size, crop_location, resize_after_crop = crop_params
        transforms.append(
            t.Crop(config.ORIGINAL_IMG_SIZE, crop_size, crop_location)
        )
        if resize_after_crop:
            transforms.append(t.Resize(config.ORIGINAL_IMG_SIZE))

    if flip_lr:
        transforms.append(t.FlipLr())

    if flip_ud:
        transforms.append(t.FlipUd())

    transforms += [
        t.AsContiguousArray(),
        ToTensor(),
        Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ]

    return Compose(transforms)

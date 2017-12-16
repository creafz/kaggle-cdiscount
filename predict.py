import os

import click
from click import option as opt
from tqdm import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmentation_transforms import make_test_augmentation_transforms
import config
import dataset
from models import get_model
from utils import load_checkpoint, save_predictions


cudnn.benchmark = config.CUDNN_BENCHMARK


@click.command()
@opt('--model-name', type=str, required=True)
@opt('--dropout-p', default=0.0)
@opt('--experiment-name', type=str, required=True)
@opt('--batch-size', default=256)
@opt('--save-npz-every-n-batches', default=256)
@opt('--flip-lr', is_flag=True)
@opt('--flip-ud', is_flag=True)
@opt('--crop', is_flag=True)
@opt('--crop-size', nargs=2, default=(160, 160))
@opt('--crop-location', type=str)
@opt('--resize-after-crop', is_flag=True)
@opt('--num-workers', default=config.NUM_WORKERS)
def main(
    model_name,
    dropout_p,
    experiment_name,
    batch_size,
    save_npz_every_n_batches,
    flip_lr,
    flip_ud,
    crop,
    crop_size,
    crop_location,
    resize_after_crop,
    num_workers,
):
    base_predictions_dir = os.path.join(
        config.PREDICTIONS_PATH,
        experiment_name,
    )
    os.makedirs(base_predictions_dir, exist_ok=True)

    predictions_dir_name = 'npz'
    if flip_lr:
        predictions_dir_name += '_fliplr'

    if flip_ud:
        predictions_dir_name += '_flipud'

    if crop:
        crop_size_width, crop_size_height = crop_size
        predictions_dir_name += (
            f'_crop_{crop_size_width}x{crop_size_height}_{crop_location}'
        )
        if resize_after_crop:
            predictions_dir_name += '_resize_after_crop'

    print(predictions_dir_name)

    predictions_dir = os.path.join(base_predictions_dir, predictions_dir_name)
    os.makedirs(predictions_dir, exist_ok=True)

    model = get_model(
       model_name,
       num_classes=config.NUM_CLASSES,
       dropout_p=dropout_p,
    )
    checkpoint = load_checkpoint(f'{experiment_name}_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda().eval()

    if not crop:
        crop_params = None
    else:
        crop_params = crop_size, crop_location, resize_after_crop

    transform = make_test_augmentation_transforms(
        crop_params,
        flip_lr,
        flip_ud,
    )
    test_dataset = dataset.CdiscountTestDataset(
        bson_filepath=config.TEST_BSON_FILE,
        transform=transform,
    )

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    saved_batch_index = 0
    predictions = []
    for batch_index, (image_groups, product_ids, img_nums) in tqdm(
        enumerate(test_data_loader),
        total=len(test_data_loader),
    ):
        images = Variable(image_groups, volatile=True).cuda()
        logits = model(images)
        probs = F.softmax(logits)
        numpy_probs = probs.cpu().data.numpy()
        predictions.append(numpy_probs)
        if batch_index != 0 and batch_index % save_npz_every_n_batches == 0:
            save_predictions(
                predictions,
                os.path.join(predictions_dir, f'{batch_index}.npz'),
            )
            saved_batch_index += 1
            predictions = []
    save_predictions(
        predictions,
        os.path.join(predictions_dir, f'{batch_index}.npz'),
    )


if __name__ == '__main__':
    main()
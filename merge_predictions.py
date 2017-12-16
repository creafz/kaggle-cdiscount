import os

import click
from click import option as opt, argument as arg
import numpy as np
from scipy.stats import gmean
from tqdm import tqdm

import config
from utils import save_predictions, npz_file_iterator


@click.command()
@arg('source-dirs', nargs=-1)
@opt('--target-dir', type=str, required=True)
@opt('--averaging', type=click.Choice(['mean', 'gmean']), default='mean')
def main(
    source_dirs,
    target_dir,
    averaging,
):
    source_dir_paths = [
        os.path.join(config.PREDICTIONS_PATH, dir_) for dir_ in source_dirs
    ]
    target_dir_path = os.path.join(config.PREDICTIONS_PATH, target_dir)
    os.makedirs(target_dir_path, exist_ok=True)

    print('source directories:\n' + '\n'.join(source_dir_paths) + '\n')
    print('target directory:\n' + target_dir_path)

    mean = np.mean if averaging == 'mean' else gmean

    dirs_filenames = [os.listdir(dir_) for dir_ in source_dir_paths]
    assert dirs_filenames
    filenames = dirs_filenames[0]
    for (other_filenames, dir_) in zip(dirs_filenames, source_dir_paths):
        assert set(filenames) == set(other_filenames)

    for filename in tqdm(filenames):
        predictions = []
        for prediction_row in zip(
            *(
                npz_file_iterator(os.path.join(dir_path, filename))
                for dir_path in source_dir_paths
            ),
        ):
            prediction = mean(np.vstack(prediction_row), axis=0)
            predictions.append(prediction)
        save_predictions(predictions, os.path.join(target_dir_path, filename))


if __name__ == '__main__':
    main()
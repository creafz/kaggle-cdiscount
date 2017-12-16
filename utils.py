from collections import defaultdict
import hashlib
import os
import pickle
import random

import numpy as np
from pycrayon import CrayonClient
from scipy.sparse import save_npz, load_npz, csr_matrix
import torch

import config


def save_pickle(data, filename, verbose=False):
    if verbose:
        print(f'Saving {filename}')
    with open(os.path.join(config.PICKLED_DATA_PATH, filename), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(os.path.join(config.PICKLED_DATA_PATH, filename), 'rb') as f:
        return pickle.load(f)


def make_path_func(base_dir):
    def path_func(filename):
        return os.path.join(base_dir, filename)
    return path_func


def save_checkpoint(state, filename, verbose=False):
    if verbose:
        print(f'Saving {filename}')
    filepath = os.path.join(config.SAVED_MODELS_PATH, filename)
    torch.save(state, filepath)
    return filepath


def load_checkpoint(filename):
    path = os.path.join(config.SAVED_MODELS_PATH, filename)
    return torch.load(path)


class MetricMonitor:

    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'sum': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, value, n=None, multiply_by_n=True):
        if n is None:
            n = self.batch_size
        metric = self.metrics[metric_name]
        if multiply_by_n:
            metric['sum'] += value * n
        else:
            metric['sum'] += value
        metric['count'] += n
        metric['avg'] = metric['sum'] / metric['count']

    def get_avg(self, metric_name):
        return self.metrics[metric_name]['avg']

    def get_metric_values(self):
        return [
            (metric, values['avg']) for metric, values in self.metrics.items()
        ]

    def __str__(self):
        return ' | '.join(
            f'{metric_name} {metric["avg"]:.6f}'
            for metric_name, metric in self.metrics.items()
        )

def make_crayon_experiments(experiment_name, new=True):
    client = CrayonClient(hostname=config.CRAYON_SERVER_HOSTNAME)
    train_experiment_name = f'{experiment_name}_train'
    valid_experiment_name = f'{experiment_name}_valid'
    if new:
        try:
            client.remove_experiment(train_experiment_name)
        except ValueError:
            pass
        try:
            client.remove_experiment(valid_experiment_name)
        except ValueError:
            pass
        train_experiment = client.create_experiment( train_experiment_name)
        train_experiment.scalar_steps['lr'] = 1
        valid_experiment = client.create_experiment(valid_experiment_name)
    else:
        train_experiment = client.open_experiment(train_experiment_name)
        valid_experiment = client.open_experiment(valid_experiment_name)
    return train_experiment, valid_experiment


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_predictions(predictions, filepath):
    arr = np.vstack(predictions)
    arr[arr < 1e-5] = 0
    matrix = csr_matrix(arr)
    save_npz(filepath, matrix)


def get_sha1_hash(file_content):
    hasher = hashlib.sha1()
    hasher.update(file_content)
    return hasher.hexdigest()


def calculate_accuracy(outputs, targets):
    _, predictions = outputs.topk(1, 1, True, True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))
    correct_k = correct[0].view(-1).float().sum(0)
    return correct_k.data.cpu()[0]


def npz_file_iterator(filepath):
    matrix = load_npz(filepath)
    for row in matrix:
        yield np.array(row.todense()).flatten()


def npz_directory_iterator(directory_path):
    filenames = sorted(
        os.listdir(directory_path),
        key=lambda path:int(path.split('.')[0]),
    )
    for filename in filenames:
        filepath = os.path.join(directory_path, filename)
        yield from npz_file_iterator(filepath)

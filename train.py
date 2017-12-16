from collections import defaultdict
import os
import shutil

import click
from click import option as opt
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import nn

from augmentation_transforms import make_augmentation_transforms
import config
import dataset
from models import get_model
from optimizer import get_optimizer
from utils import (
    save_checkpoint,
    load_checkpoint,
    MetricMonitor,
    make_crayon_experiments,
    set_seed,
    calculate_accuracy,
)


cudnn.benchmark = config.CUDNN_BENCHMARK


def forward_pass(
    images,
    targets,
    model,
    loss_fn,
    epoch,
    stream,
    monitor,
    mode='train',
):
    volatile = mode != 'train'
    images = Variable(images, volatile=volatile).cuda(async=True)
    targets = Variable(targets, volatile=volatile).long().cuda(async=True)
    outputs = model(images)
    accuracy = calculate_accuracy(outputs, targets)
    monitor.update('accuracy', accuracy, multiply_by_n=False)
    loss = loss_fn(outputs, targets)
    monitor.update('loss', loss.data[0])
    stream.set_description(f'epoch: {epoch} | {mode}: {monitor}')
    return loss, outputs


def train(
    train_data_loader,
    model,
    optimizer,
    iter_size,
    loss_fn,
    epoch,
    experiment,
    experiment_name,
):
    model.train()
    train_monitor = MetricMonitor(batch_size=train_data_loader.batch_size)
    stream = tqdm(train_data_loader)
    for i, (images, targets) in enumerate(stream, start=1):
        loss, _ = forward_pass(
            images,
            targets,
            model,
            loss_fn,
            epoch,
            stream,
            train_monitor,
            mode='train',
        )
        loss.backward()

        if i % iter_size == 0 or i == len(train_data_loader):
            optimizer.step()
            optimizer.zero_grad()

        if i % config.SAVE_EACH_ITERATIONS == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'{experiment_name}_intermediate.pth',
                verbose=True,
            )
    experiment.add_scalar_value(
        'optimizer/lr',
        optimizer.param_groups[0]['lr'],
        step=epoch,
    )
    for metric, value in train_monitor.get_metric_values():
        experiment.add_scalar_value(f'metric/{metric}', value, step=epoch)


def validate(valid_data_loader, model, loss_fn, epoch, experiment):
    model.eval()
    valid_monitor = MetricMonitor(batch_size=valid_data_loader.batch_size)
    stream = tqdm(valid_data_loader)
    for images, targets in stream:
        _, outputs = forward_pass(
            images,
            targets,
            model,
            loss_fn,
            epoch,
            stream,
            valid_monitor,
            mode='valid',
        )
    for metric, value in valid_monitor.get_metric_values():
        experiment.add_scalar_value(f'metric/{metric}', value, step=epoch)
    return valid_monitor


def train_and_validate(
    train_data_loader,
    valid_data_loader,
    model,
    optimizer,
    iter_size,
    scheduler,
    loss_fn,
    epochs,
    experiment_name,
    experiments,
    start_epoch,
    best_val_loss,
):
    train_experiment, valid_experiment = experiments
    if best_val_loss is None:
        best_val_loss = float('+inf')

    for epoch in range(start_epoch, epochs + 1):
        train(
            train_data_loader,
            model,
            optimizer,
            iter_size,
            loss_fn,
            epoch,
            train_experiment,
            experiment_name,
        )
        val_monitor = validate(
            valid_data_loader,
            model,
            loss_fn,
            epoch,
            valid_experiment,
        )
        val_loss = val_monitor.get_avg('loss')
        checkpoint_path = save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            },
            f'{experiment_name}_{epoch}_{val_loss}.pth',
            verbose=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                config.SAVED_MODELS_PATH,
                f'{experiment_name}_best.pth',
            )
            shutil.copyfile(checkpoint_path, best_model_path)

        scheduler.step()

    return model


@click.command()
@opt('--batch-size', default=1)
@opt('--optimizer-name', type=click.Choice(['adam', 'sgd']), default='sgd')
@opt('--lr', default=0.01)
@opt('--epochs', default=100)
@opt('--iter-size', default=1)
@opt('--model-name', type=str, required=True)
@opt('--experiment-name', type=str, required=True)
@opt('--load-best-model', is_flag=True)
@opt('--load-best-model-optimizer', is_flag=True)
@opt('--start-epoch', default=1)
@opt('--create-new-experiment', is_flag=True)
@opt('--seed', default=config.SEED)
@opt('--freeze-model', is_flag=True)
@opt('--dropout-p', default=0.0)
@opt('--num-workers', default=config.NUM_WORKERS)
@opt('--img-size', nargs=2, default=config.ORIGINAL_IMG_SIZE)
def main(
    batch_size,
    optimizer_name,
    lr,
    epochs,
    iter_size,
    model_name,
    experiment_name,
    load_best_model,
    load_best_model_optimizer,
    start_epoch,
    create_new_experiment,
    seed,
    freeze_model,
    num_workers,
    dropout_p,
    img_size
):
    set_seed(seed)
    transform_train = make_augmentation_transforms(
        img_size,
        mode='train',
    )
    transform_valid = make_augmentation_transforms(
        img_size,
        mode='valid',
    )

    full_experiment_name = f'{experiment_name}'
    print(full_experiment_name)

    model = get_model(
        model_name,
        freeze=freeze_model,
        num_classes=config.NUM_CLASSES,
        dropout_p=dropout_p,
    )
    best_val_loss = None
    model = model.cuda()
    optimizer = get_optimizer(optimizer_name, lr, model)
    if load_best_model:
        checkpoint_filename = f'{full_experiment_name}_best.pth'
        print(f'Loading checkpoint {checkpoint_filename}')
        checkpoint = load_checkpoint(checkpoint_filename)
        model.load_state_dict(checkpoint['state_dict'])
        best_val_loss = checkpoint.get('val_loss')

        if load_best_model_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.state = defaultdict(dict, optimizer.state)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    experiments = make_crayon_experiments(
        full_experiment_name,
        new=not load_best_model or create_new_experiment,
    )
    loss_fn = nn.CrossEntropyLoss().cuda()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    bson_filepaths = {
        'train.bson': config.TRAIN_BSON_FILE,
        'test.bson': config.TEST_BSON_FILE,
    }
    train_dataset = dataset.CdiscountTrainDataset(
        bson_filepaths=bson_filepaths,
        mode='train',
        transform=transform_train,
    )
    valid_dataset = dataset.CdiscountTrainDataset(
        bson_filepaths=bson_filepaths,
        mode='valid',
        transform=transform_valid,
    )

    data_loader_args = {
        'pin_memory': True,
        'num_workers': num_workers,
    }

    train_data_loader = DataLoader(
        **data_loader_args,
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_data_loader = DataLoader(
        **data_loader_args,
        dataset=valid_dataset,
        batch_size=batch_size,
    )

    train_and_validate(
        train_data_loader,
        valid_data_loader,
        model,
        optimizer,
        iter_size,
        scheduler,
        loss_fn,
        epochs,
        full_experiment_name,
        experiments,
        start_epoch,
        best_val_loss,
    )


if __name__ == '__main__':
    main()

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models

import pretrainedmodels


class TorchvisionModelWrapper(nn.Module):
    """
    Supports ResNet and DenseNet models from https://github.com/pytorch/vision
    """

    def __init__(
        self,
        model_name,
        num_classes,
        dropout_p=None,
        pretrained=True,
        freeze=False,
    ):
        super().__init__()
        try:
            model = getattr(models, model_name)(pretrained=pretrained)
        except AttributeError:
            raise Exception(f'Unknown torchvision model {model_name}')

        self.model_name = model_name
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p else None

        if model_name.startswith('resnet'):
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Linear(model.fc.in_features, num_classes)
        elif model_name.startswith('densenet'):
            self.features = nn.Sequential(
                *model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(
                model.classifier.in_features,
                num_classes,
            )
        else:
            raise Exception('Only ResNet and DenseNet models are supported')

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PretrainedModelsWrapper(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        dropout_p=None,
        pretrained=True,
        freeze=False,
    ):
        """
        Supports ResNeXt and InceptionResNetV2 models from
        https://github.com/Cadene/pretrained-models.pytorch
        """
        super().__init__()
        try:
            model_factory = getattr(pretrainedmodels, model_name)
            model = model_factory(
                num_classes=1000,
                pretrained='imagenet' if pretrained else None,
            )
        except AttributeError:
            raise Exception(f'Unknown pretrained model {model_name}')

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p else None
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if model_name == 'inceptionresnetv2':
            self.features = nn.Sequential(*list(model.children())[:-2])
            fc_layer = model.last_linear
        elif model_name.startswith('resnext'):
            self.features = model.features
            fc_layer = model.last_linear
        else:
            raise Exception(
                'Only ResNeXt and InceptionResNetV2 '
                'models models are supported'
            )

        self.classifier = nn.Linear(fc_layer.in_features, num_classes)
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name, **params):
    if model_name.startswith(('resnet', 'densenet')):
        wrapper = TorchvisionModelWrapper
    elif model_name.startswith('resnext') or model_name == 'inceptionresnetv2':
        wrapper = PretrainedModelsWrapper
    else:
        raise Exception(f'Unknown model_name {model_name}')
    model = wrapper(model_name, **params)
    return model


def test_original_input(
    wrapper,
    models,
    model_name,
    pretrained,
    classes,
    num_channels,
    batch_size,
    **kw,
):
    img_size = (224, 224)
    original_model = getattr(models, model_name)(pretrained=pretrained, **kw)
    wrapped_model = wrapper(model_name, classes)
    input_var = Variable(torch.randn(batch_size, num_channels, *img_size))
    if model_name.startswith('resnet'):
        original_output = nn.Sequential(
            *list(original_model.children())[:-1]
        )(input_var)
    elif model_name.startswith('densenet'):
        original_output = nn.Sequential(
            *original_model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )(input_var)
    elif model_name.startswith('resnext'):
        original_output = original_model.features(input_var)
    elif model_name == 'inceptionresnetv2':
        original_output = nn.Sequential(
            *list(original_model.children())[:-2]
        )(input_var)
    else:
        raise Exception(f'Unknown model_name {model_name}')
    wrapped_output = wrapped_model.features(input_var)
    assert np.all(
        np.isclose(original_output.data.numpy(), wrapped_output.data.numpy())
    )


def test_input_with_another_size(
    wrapper,
    model_name,
    classes,
    num_channels,
    batch_size,
):
    img_size = (250, 250)
    model = wrapper(model_name, classes)
    input_var = Variable(torch.randn(batch_size, num_channels, *img_size))
    output = model(input_var)
    assert output.size() == torch.Size([batch_size, classes])


if __name__ == '__main__':
    test_args = {
        'classes': 5,
        'num_channels': 3,
        'batch_size': 4,
    }

    for model_name in (
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
        'densenet121',
        'densenet169',
        'densenet201',
        'densenet161',
    ):
        test_original_input(
            TorchvisionModelWrapper,
            models,
            model_name,
            pretrained=True,
            **test_args,
        )
        test_input_with_another_size(
            TorchvisionModelWrapper,
            model_name,
            **test_args,
        )

    for model_name in (
        'resnext101_32x4d',
        'resnext101_64x4d',
        'inceptionresnetv2',
    ):
        test_original_input(
            PretrainedModelsWrapper,
            pretrainedmodels,
            model_name,
            pretrained='imagenet',
            num_classes=1000,
            **test_args,
        )
        test_input_with_another_size(
            PretrainedModelsWrapper,
            model_name,
            **test_args,
        )

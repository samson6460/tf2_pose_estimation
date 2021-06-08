"""ResU-Net model.
"""

from functools import reduce

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Add
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.applications import ResNet101

from .custom_layers import Softmax

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def Conv2D_BN_Leaky(*args, **kwargs):
    """Convolution2D followed by BatchNormalization and LeakyReLU."""
    conv_kwargs = {
        'use_bias': False,
        'padding': 'same',
        'kernel_initializer':'he_normal'}
    conv_kwargs.update(kwargs)
    return compose(
        Conv2D(*args, **conv_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def Conv2DTranspose_BN_Leaky(*args, **kwargs):
    """Transpose Convolution2D followed by BatchNormalization and LeakyReLU."""
    convtrans_kwargs = {
        'use_bias': False,
        'padding': 'same',
        'kernel_initializer':'he_normal'}
    convtrans_kwargs.update(kwargs)
    return compose(
        Conv2DTranspose(*args, **convtrans_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def UpConv2D_BN_Leaky(*args, **kwargs):
    """Transpose Convolution2D followed by BatchNormalization and LeakyReLU."""
    conv_kwargs = {
        'use_bias': False,
        'padding': 'same',
        'kernel_initializer':'he_normal'}
    conv_kwargs.update(kwargs)
    return compose(
        UpSampling2D(size = (2, 2)),
        Conv2D(*args, **conv_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def up_resblock_module(x, skip_connect, num_filters, num_blocks):
    y = UpConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = compose(UpConv2D_BN_Leaky(num_filters//4, (1, 1)),
                Conv2D_BN_Leaky(num_filters//4, (3, 3)),
                Conv2D_BN_Leaky(num_filters, (1, 1)))(x)
    x = Add()([x, y])
    x = Concatenate()([x, skip_connect])
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)

    for _ in range(num_blocks):
        y = compose(
            Conv2D_BN_Leaky(num_filters//4, (1, 1)),
            Conv2D_BN_Leaky(num_filters//4, (3, 3)),
            Conv2D_BN_Leaky(num_filters, (1, 1)))(x)
        x = Add()([x, y])
    return x


def resunet(resnet_func=ResNet101,
            input_shape=(416, 416, 3),
            pretrained_backbone="imagenet",
            pretrained_weights=None,
            upskip_id=[-33, 80, 38, 4],
            res_num_blocks=[23, 4, 3, 1],
            num_points=15,
            activation='sigmoid'):
    """Create ResU-Net architecture.
    
    Args:
        resnet_func: A Resnet from
            tensorflow.keras.applications.
            e.g., tensorflow.keras.applications.ResNet101.
        input_shape: A tuple of 3 integers,
            shape of input image.
        pretrained_backbone: one of None (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        pretrained_weights: A string, 
            file path of pretrained model.
        upskip_id: A list of integer,
            index of skip connections from extracting path.
        res_num_blocks: A list of integer.
            number of repetitions of up-residual blocks.
        num_points: An integer,
            number of keypoints.
        activation: A string or None,
            activation to add to the top of the network.
            One of "sigmoid"、"softmax"(per channel) or None.

    Returns:
        A tf.keras Model.
    """
    if pretrained_weights is not None:
        pretrained_backbone = None
    
    appnet = resnet_func(
        include_top=False,
        weights=pretrained_backbone,
        input_shape=input_shape)

    x = appnet.output
    num_filters = x.shape[-1]

    for id, num_blocks in zip(upskip_id, res_num_blocks):
        num_filters //= 2
        x = up_resblock_module(x, appnet.layers[id].output,
            num_filters, num_blocks)

    x = UpConv2D_BN_Leaky(32, (3, 3))(x)
    x = Concatenate()([x, appnet.layers[0].output])
    x = Conv2D_BN_Leaky(32, (3, 3))(x)
    x = Conv2D(num_points, 1)(x)

    if activation=="sigmoid":
        outputs = Activation("sigmoid")(x)
    elif activation=="softmax":
        outputs = Softmax(axis=(1, 2))(x)
    else:
        outputs = x

    model = Model(appnet.input, outputs)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model
"""Custom layers.
"""

from functools import reduce

from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D


class Softmax(Layer):
    def __init__(self, axis):
        super(Softmax, self).__init__()
        self.axis = axis

    def call(self, inputs):
        return softmax(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis 
        })
        return config


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
    """Up Convolution2D followed by BatchNormalization and LeakyReLU."""
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


def Conv2DTranspose_BN_ReLU(*args, **kwargs):
    """Transpose Convolution2D followed by BatchNormalization and ReLU."""
    convtrans_kwargs = {
        'use_bias': False,
        'padding': 'same',
        'kernel_initializer':'he_normal'}
    convtrans_kwargs.update(kwargs)
    return compose(
        Conv2DTranspose(*args, **convtrans_kwargs),
        BatchNormalization(),
        ReLU())

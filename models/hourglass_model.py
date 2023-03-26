"""Stacked hourglass model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Add
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation

from .custom_layers import Softmax
from .custom_layers import Conv2D_BN_Leaky


def Conv2D_BN_Leaky(*args, **kwargs):
    """Convolution2D followed by BatchNormalization and ReLU."""
    conv_kwargs = {
        'use_bias': True,
        'padding': 'same',
        'kernel_initializer':'he_normal'}
    conv_kwargs.update(kwargs)
    return Conv2D_BN_Leaky(*args, **kwargs)


def resblock_module(x, num_filters):
    skip = Conv2D_BN_Leaky(num_filters, (1, 1))(x)

    x = Conv2D_BN_Leaky(num_filters//2, (1, 1))(x)
    x = Conv2D_BN_Leaky(num_filters//2, (3, 3), padding='same')(x)
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = Add()([skip, x])

    return x


def hourglass_module(input_tensor, stage):
    stage -= 1
    skip = resblock_module(input_tensor, 256)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(input_tensor)
    x = resblock_module(x, 256)
    if stage == 0:
        x = resblock_module(x, 256)
    else:
        x = hourglass_module(x, stage)
    x = resblock_module(x, 256)
    x = UpSampling2D(2)(x)
    x = Add()([skip, x])

    return x


def front_module(input_tensor, num_filters=256):
    x = Conv2D_BN_Leaky(
        num_filters//4, (7, 7), (2, 2),
        padding='same')(input_tensor)
    x = resblock_module(x, num_filters//2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = resblock_module(x, num_filters//2)
    x = resblock_module(x, num_filters)

    return x

def stack_module(input_tensor, num_points,
                 num_filters=256, stage=4,
                 activation="sigmoid", is_head=False):
    x = hourglass_module(input_tensor, stage)
    x = resblock_module(x, num_filters)
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)
    outputs = Conv2D(num_points, (1, 1))(x)
    if activation=="sigmoid":
        outputs = Activation("sigmoid")(outputs)
    elif activation=="softmax":
        outputs = Softmax(axis=(1, 2))(outputs)
    
    if is_head:
        return outputs
    else:
        x = Conv2D(num_filters, (1, 1))(x)
        skip = Conv2D(num_filters, (1, 1))(outputs)
        x = Add()([skip, x])
        return outputs, x


def stack_hourglass_net(
    input_shape=(512, 512, 3),
    num_stacks=8, num_points=15,
    num_filters=256, stage=4,
    activation="sigmoid",
    pretrained_weights=None):
    """Create stacked hourglass network architecture.

    Args:
        input_shape: A tuple of 3 integers,
            shape of input image.
        num_stacks: An integer,
            number of stacks of hourglass network.
        num_points: An integer,
            number of keypoints.
        num_filters: An integer,
            number of convolution filters.
        stage: An integer,
            stage of each hourglass module.
        activation: A string or None,
            activation to add to the top of the network.
            One of "sigmoid"、"softmax"(per channel) or None.
        pretrained_weights: A string, 
            file path of pretrained model.
    
    Returns:
        A tf.keras Model.
    """
    output_list = []
    inputs = Input(input_shape)
    x = front_module(inputs, num_filters=num_filters)

    for _ in range(num_stacks - 1):
        skip = x
        outputs, x = stack_module(
            x, num_points,
            num_filters=num_filters,
            stage=stage,
            activation=activation)
        x = Add()([skip, x])
        output_list.append(outputs)

    outputs = stack_module(
        x, num_points,
        num_filters=num_filters,
        stage=stage,
        activation=activation,
        is_head=True)
    output_list.append(outputs)

    model = Model(inputs, output_list)

    if pretrained_weights is not None:    
        model.load_weights(pretrained_weights)
    
    return model
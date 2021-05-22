from functools import wraps
from functools import reduce

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input


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
    """Convolution2D followed by BatchNormalization and ReLU."""
    return compose(
        Conv2D(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_module(x, num_filters):
    skip = Conv2D_BN_Leaky(num_filters, (1, 1))(x)

    x = Conv2D_BN_Leaky(num_filters//2, (1, 1))(x)
    x = Conv2D_BN_Leaky(num_filters//2, (3, 3), padding='same')(x)
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = Add()([skip, x])

    return x


def hourglass_module(input_tensor, num_classes):
    num_classes -= 1
    skip = resblock_module(input_tensor, 256)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(input_tensor)
    x = resblock_module(x, 256)
    if num_classes == 0:
        x = resblock_module(x, 256)
    else:
        x = hourglass_module(x, num_classes)
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
                 num_filters=256, num_classes=4, is_head=False):
    x = hourglass_module(input_tensor, num_classes)
    x = resblock_module(x, num_filters)
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)
    outputs = Conv2D(num_points, (1, 1), activation="sigmoid")(x)

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
    num_filters=256, num_classes=4,
    pretrained_weights=None):
    """Create stacked hourglass network architecture.

    Args:
        input_shape: A tuple of 3 integers,
            shape of input image.
        num_stacks: A integer.
        num_points: A integer,
            number of key points.
        num_filters: A integer,
            number of convolution filters.
        num_classes: A integer,
            number of hourglass module classes.
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
            num_classes=num_classes)
        x = Add()([skip, x])
        output_list.append(outputs)

    outputs = stack_module(
        x, num_points,
        num_filters=num_filters,
        num_classes=num_classes, is_head=True)
    output_list.append(outputs)

    model = Model(inputs, output_list)

    if pretrained_weights is not None:    
        model.load_weights(pretrained_weights)
    
    return model
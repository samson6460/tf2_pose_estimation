"""ResUNet model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.applications import ResNet101

from .custom_layers import Softmax, compose
from .custom_layers import Conv2D_BN_Leaky, UpConv2D_BN_Leaky


def up_resblock_module(tensor, skip_connect, num_filters, num_blocks):
    """Up residual block module."""
    skip_tensor = UpConv2D_BN_Leaky(num_filters, (1, 1))(tensor)
    tensor = compose(
        UpConv2D_BN_Leaky(num_filters//4, (1, 1)),
        Conv2D_BN_Leaky(num_filters//4, (3, 3)),
        Conv2D_BN_Leaky(num_filters, (1, 1)))(tensor)
    tensor = Add()([tensor, skip_tensor])
    tensor = Concatenate()([tensor, skip_connect])
    tensor = Conv2D_BN_Leaky(num_filters, (1, 1))(tensor)

    for _ in range(num_blocks):
        skip_tensor = compose(
            Conv2D_BN_Leaky(num_filters//4, (1, 1)),
            Conv2D_BN_Leaky(num_filters//4, (3, 3)),
            Conv2D_BN_Leaky(num_filters, (1, 1)))(tensor)
        tensor = Add()([tensor, skip_tensor])
    return tensor


def resunet(resnet_func=ResNet101,
            input_shape=(512, 512, 3),
            pretrained_backbone="imagenet",
            pretrained_weights=None,
            upskip_id=[-33, 80, 38, 4],
            res_num_blocks=[2, 22, 3, 2],
            skip_connect_input=True,
            num_points=15,
            activation='sigmoid'):
    """Create ResUNet architecture.
    
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
        skip_connect_input: A boolean,
            whether to skip connect to input tensor.
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

    tensor = appnet.output
    num_filters = tensor.shape[-1]

    for i_skip, num_blocks in zip(upskip_id, res_num_blocks):
        num_filters //= 2
        tensor = up_resblock_module(
            tensor, appnet.layers[i_skip].output,
            num_filters, num_blocks)

    if skip_connect_input:
        tensor = UpConv2D_BN_Leaky(32, (3, 3))(tensor)
        tensor = Concatenate()([tensor, appnet.layers[0].output])
        tensor = Conv2D_BN_Leaky(32, (3, 3))(tensor)

    tensor = Conv2D(num_points, 1)(tensor)

    if activation=="sigmoid":
        outputs = Activation("sigmoid")(tensor)
    elif activation=="softmax":
        outputs = Softmax(axis=(1, 2))(tensor)
    else:
        outputs = tensor

    model = Model(appnet.input, outputs)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model

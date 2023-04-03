"""Simple baseline model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.applications import ResNet50

from .custom_layers import Softmax
from .custom_layers import Conv2DTranspose_BN_ReLU

def pose_resnet(resnet_func=ResNet50,
                input_shape=(512, 512, 3),
                pretrained_backbone="imagenet",
                pretrained_weights=None,
                num_layers=3,
                num_filters=256,
                kernel_size=4,
                num_points=15,
                activation='sigmoid'):
    """Create Simple Baseline Model architecture.
    
    Args:
        resnet_func: A Resnet from
            tensorflow.keras.applications.
            e.g., tensorflow.keras.applications.ResNet50.
        input_shape: A tuple of 3 integers,
            shape of input image.
        pretrained_backbone: one of None (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        pretrained_weights: A string, 
            file path of pretrained model.
        num_layers: An integer, number of transpose convolution layers.
            This arg will effect the output shape.
        num_filters: An integer,
            number of filters of each transpose convolution.
        kernel_size: An integer,
            kernel size of each transpose convolution.
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

    for _ in range(num_layers):
        tensor = Conv2DTranspose_BN_ReLU(
            num_filters, kernel_size, strides=2)(tensor)

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

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
    if pretrained_weights is not None:
        pretrained_backbone = None
    
    appnet = resnet_func(
        include_top=False,
        weights=pretrained_backbone,
        input_shape=input_shape)

    x = appnet.output

    for _ in range(num_layers):
        x = Conv2DTranspose_BN_ReLU(
            num_filters, kernel_size, strides=2)(x)

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
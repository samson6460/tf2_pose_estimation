# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Unet model.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import Activation

from .custom_layers import Softmax


def Conv2D_Acti_BN(input_tensor, activation, *args):
    output_tensor = Conv2D(*args,
                           activation=activation,
                           padding='same',
                           kernel_initializer='he_normal')(input_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def UpConv2D_Acti_BN(input_tensor, activation, *args):
    output_tensor = UpSampling2D(size = (2,2))(input_tensor)
    output_tensor = Conv2D(*args,
                           activation=activation,
                           padding='same',
                           kernel_initializer='he_normal')(output_tensor)
    output_tensor = BN()(output_tensor)
    return output_tensor


def unet(pretrained_weights=None,
         input_shape=(512, 512, 3),
         conv_activation='relu',
         num_points=15,
         activation='sigmoid'):
    """Create U-Net network architecture.
    
    Args:
        pretrained_weights: A string, 
            file path of pretrained model.
        input_shape: A tuple of 3 integers,
            shape of input image.
        conv_activation: A string,
            activation function for convolutional layer.
        num_points: An integer,
            number of keypoints.
        activation: A string or None,
            activation to add to the top of the network.
            One of "sigmoid"、"softmax"(per channel) or None.


    Returns:
        A tf.keras Model.
    """
    inputs = Input(input_shape)
    conv1 = Conv2D_Acti_BN(inputs, conv_activation, 64, 3)
    conv1 = Conv2D_Acti_BN(conv1, conv_activation, 64, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_Acti_BN(pool1, conv_activation, 128, 3)
    conv2 = Conv2D_Acti_BN(conv2, conv_activation, 128, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_Acti_BN(pool2, conv_activation, 256, 3)
    conv3 = Conv2D_Acti_BN(conv3, conv_activation, 256, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_Acti_BN(pool3, conv_activation, 512, 3)
    conv4 = Conv2D_Acti_BN(conv4, conv_activation, 512, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_Acti_BN(pool4, conv_activation, 1024, 3)
    conv5 = Conv2D_Acti_BN(conv5, conv_activation, 1024, 3)

    up6 = UpConv2D_Acti_BN(conv5, conv_activation, 512, 2)
    merge6 = concatenate([conv4, up6], axis = 3)
    conv6 = Conv2D_Acti_BN(merge6, conv_activation, 512, 3)
    conv6 = Conv2D_Acti_BN(conv6, conv_activation, 512, 3)

    up7 = UpConv2D_Acti_BN(conv6, conv_activation, 256, 2)
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D_Acti_BN(merge7, conv_activation, 256, 3)
    conv7 = Conv2D_Acti_BN(conv7, conv_activation, 256, 3)

    up8 = UpConv2D_Acti_BN(conv7, conv_activation, 128, 2)
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D_Acti_BN(merge8, conv_activation, 128, 3)
    conv8 = Conv2D_Acti_BN(conv8, conv_activation, 128, 3)

    up9 = UpConv2D_Acti_BN(conv8, conv_activation, 64, 2)
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D_Acti_BN(merge9, conv_activation, 64, 3)
    conv9 = Conv2D_Acti_BN(conv9, conv_activation, 64, 3)
    conv10 = Conv2D(num_points, 1)(conv9)

    if activation=="sigmoid":
        outputs = Activation("sigmoid")(conv10)
    elif activation=="softmax":
        outputs = Softmax(axis=(1, 2))(conv10)
    else:
        outputs = conv10

    model = Model(inputs, outputs)
    
    if pretrained_weights is not None:   
        model.load_weights(pretrained_weights)

    return model
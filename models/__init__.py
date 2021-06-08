# Copyright 2021 Samson Woof. All Rights Reserved.
# =============================================================================

"""Keypoint detection models.
"""

from .unet_model import unet
from .deep_lab_model import deeplabv3
from .hourglass_model import stack_hourglass_net
from .resunet_model import resunet
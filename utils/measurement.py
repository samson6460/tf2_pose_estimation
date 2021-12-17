# Copyright 2021 Samson Woof. All Rights Reserved.
# =============================================================================

"""Measurement tools for keypoint detection.
"""

import numpy as np
import pandas as pd
from math import log

from .tools import heatmap2point


def get_score_table(ground_truth, prediction,
                    decode_method="max",
                    class_names=["Head", "Neck", "Hip",
                        "L_shoulder", "L_elbow", "L_wrist",
                        "R_shoulder", "R_elbow", "R_wrist",
                        "L_hip", "L_knee", "L_ankle",
                        "R_hip", "R_knee", "R_ankle"],
                    dist_thresh=None,
                    oks_thresh=0.5,
                    norm_index=[[3, 6], [0, 2]],
                    index=None):
    """Get mOKS and recall table.

    Args:
        ground_truth: A ndarray,
            shape should be: (None, heights, widths, num_classes).
        prediction: A ndarray,
            shape should be: (None, heights, widths, num_classes)
            or (1, heights, widths, num_classes).
            If batch size is 1, `index` is necessary.
        decode_method: One of "max" and "mean".
            "max": Use the brightest position
                of the heatmap as the keypoint.
            "mean": Use the average brightness
                of the heatmap as the keypoint.
        class_names: A list of string,   
            corresponding name of label.
        dist_thresh: None or a float,
            threshold of normalized distance for 0.5 OKS.
            If dist_thresh set as None, it'll be set as 2.35*std
            (from `ground_truth`).
            If dist_thresh set as None and batch size of
            ground_truth is 1, it'll be set as 0.1.
        oks_thresh: A float, threshold of OKS for recall.
        norm_index: A 2D array like.
        index: An integer,
            Calculate OKS of a single index.

    Return:
        A pandas.Dataframe.
    """
    if (prediction.shape[0] == 1
        and index is None
        and ground_truth.shape[0] > 1):
        raise ValueError(("batch size of prediction is 1,"
            " `index` is necessary."))
    if ground_truth.shape[0] == 1 and dist_thresh is None:
        dist_thresh = 0.1

    x_arr_true, y_arr_true = heatmap2point(
        ground_truth, method=decode_method)

    x_arr_pred, y_arr_pred = heatmap2point(
        prediction, method=decode_method)

    if index is not None:
        x_arr_true = x_arr_true[index:index + 1]
        y_arr_true = y_arr_true[index:index + 1]
        OKS_name = "OKS"
    else:
        OKS_name = "mOKS"

    dist_square = ((x_arr_true - x_arr_pred)**2
                  +(y_arr_true - y_arr_pred)**2)

    (norm_x_start, norm_x_end), (norm_y_start, norm_y_end) = norm_index

    s_x = np.sqrt(
        (x_arr_true[:, norm_x_start] - x_arr_true[:, norm_x_end])**2
        +(y_arr_true[:, norm_x_start] - y_arr_true[:, norm_x_end])**2)
    s_y = np.sqrt(
        (x_arr_true[:, norm_y_start] - x_arr_true[:, norm_y_end])**2
        +(y_arr_true[:, norm_y_start] - y_arr_true[:, norm_y_end])**2)

    norm = np.expand_dims(s_x*s_y, axis=1)

    if dist_thresh is None:
        mu_x = x_arr_true.mean(axis=0)
        mu_y = y_arr_true.mean(axis=0)

        var = (((x_arr_true-mu_x)**2
                + (y_arr_true-mu_y)**2)/norm).mean(axis=0)
        k_square = np.expand_dims(4*var, axis=0)
    else:
        k_square = - 1/(2*log(0.5))*dist_thresh**2
    oks = np.exp(-1*(dist_square)/(2*norm*k_square))
    oks_mean = oks.mean(axis=0)
    recall_arr = (oks >= oks_thresh).sum(axis=0)/len(oks)

    score_table = pd.DataFrame([oks_mean, recall_arr])
    score_table.columns = class_names
    score_table.index = [OKS_name, "recall"]

    return score_table
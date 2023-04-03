# Copyright 2021 Samson Woof. All Rights Reserved.
# =============================================================================

"""Measurement tools for keypoint detection.
"""

import numpy as np
import pandas as pd
from math import log

from .tools import heatmap2point


def _index_to_point(x_arr, y_arr, index):
    if isinstance(index, int):
        point_x = x_arr[:, index]
        point_y = y_arr[:, index]
    else:
        point_x = x_arr[:, index[0]]**2
        point_y = y_arr[:, index[0]]**2
        for idx in index[1:]:
            point_x = point_x + x_arr[:, idx]**2
            point_y = point_y + y_arr[:, idx]**2
        point_x = np.sqrt(point_x)
        point_y = np.sqrt(point_y)
    return point_x, point_y


def _get_dist(x_arr, y_arr, start, end):
    start_x, start_y = _index_to_point(x_arr, y_arr, start)
    end_x, end_y = _index_to_point(x_arr, y_arr, end)
    s_d = np.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
    return s_d


def get_score_table(ground_truth, prediction,
                    decode_method="max",
                    class_names=["Head", "Eye_L", "Eye_R", "Nose",
                                 "Upper_Lip", "Lower_Lip", "Shoulder_R","Shoulder_L",
                                 "Elbow_R", "Elbow_L", "Wrist_R", "Wrist_L",
                                 "MP_joint_R", "MP_joint_L", "Hip_R", "Hip_L",
                                 "Knee_R", "Knee_L", "Ankle_R", "Ankle_L",
                                 "MTP_joint_R", "MTP_joint_L"],
                    dist_thresh=None,
                    oks_thresh=0.5,
                    norm_index=[[6, 7], [0, [14, 15]]],
                    index=None):
    """Get mOKS(Object Keypoint Similarity) and
    PCK(Percentage of Correct Keypoints) table.

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

    point_true = heatmap2point(
        ground_truth, method=decode_method)
    x_arr_true = point_true[..., 0]
    y_arr_true = point_true[..., 1]

    point_pred = heatmap2point(
        prediction, method=decode_method)
    x_arr_pred = point_pred[..., 0]
    y_arr_pred = point_pred[..., 1]

    if index is not None:
        x_arr_true = x_arr_true[index:index + 1]
        y_arr_true = y_arr_true[index:index + 1]
        oks_name = "OKS"
    else:
        oks_name = "mOKS"

    dist_square = ((x_arr_true - x_arr_pred)**2
                  +(y_arr_true - y_arr_pred)**2)

    (norm_x_start, norm_x_end), (norm_y_start, norm_y_end) = norm_index

    s_x = _get_dist(x_arr_true, y_arr_true, norm_x_start, norm_x_end)
    s_y = _get_dist(x_arr_true, y_arr_true, norm_y_start, norm_y_end)

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
    pck_arr = (oks >= oks_thresh).sum(axis=0)/len(oks)

    score_table = pd.DataFrame([oks_mean, pck_arr])
    score_table.columns = class_names
    score_table.index = [oks_name, "PCK"]

    return score_table

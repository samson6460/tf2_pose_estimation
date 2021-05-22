# Copyright 2021 Samson Woof. All Rights Reserved.
# =============================================================================

"""Measurement tools for keypoint detection.
"""

import numpy as np
import pandas as pd


def get_score_table(ground_truth, prediction,
                    label_class=["Head", "Neck", "Hip",
                        "L_shoulder", "L_elbow", "L_wrist",
                        "R_shoulder", "R_elbow", "R_wrist",
                        "L_hip", "L_knee", "L_ankle",
                        "R_hip", "R_knee", "R_ankle"],
                    oks_thresh=0.5,
                    norm_index=[[3, 6], [0, 2]],
                    index=None,
                    use_std=True):
    """Get mOKS and recall table.

    Args:
        ground_truth: A ndarray,
            shape should be: (None, heights, widths, num_classes).
        prediction: A ndarray,
            shape should be: (None, heights, widths, num_classes)
            or (1, heights, widths, num_classes).
            If batch size is 1, `index` is necessary.
        label_class: A list of string,   
            corresponding name of label.
        thresh: A float.
        norm_index: A 2D array like.
        index: An integer,
            Calculate OKS of a single index.
        use_std: A boolean,
            whether to use standard deviation to control fall off.
            If the batch size of ground_truth is 1,
            this argument will be forced to False.

    Return:
        A pandas.Dataframe.
    """
    if (prediction.shape[0] == 1
        and index is None
        and ground_truth.shape[0] > 1):
        raise ValueError(("batch size of prediction is 1,"
            " `index` is necessary."))
    if ground_truth.shape[0] == 1:
        use_std = False

    class_num = ground_truth.shape[-1]
    height = ground_truth.shape[1]
    width =  ground_truth.shape[2]
    area = height*width

    max_index_true = ground_truth.reshape(
        -1, area, class_num).argmax(axis=1)

    x_arr_true = max_index_true%height
    y_arr_true = max_index_true//height
    x_arr_true_all = x_arr_true
    y_arr_true_all = y_arr_true

    if index is not None:
        x_arr_true = x_arr_true[index:index + 1]
        y_arr_true = y_arr_true[index:index + 1]
        OKS_name = "OKS"
    else:
        OKS_name = "mOKS"

    max_index_pred = prediction.reshape(
        -1, area, class_num).argmax(axis=1)

    x_arr_pred = max_index_pred%height
    y_arr_pred = max_index_pred//height

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

    if use_std:
        mu_x = x_arr_true_all.mean(axis=0)
        mu_y = y_arr_true_all.mean(axis=0)

        var = (((x_arr_true_all-mu_x)**2
                + (y_arr_true_all-mu_y)**2)/norm).mean(axis=0)
        k_square = np.expand_dims(4*var, axis=0)
    else:
        k_square = 1

    oks = np.exp(-1*(dist_square)/(2*norm*k_square))
    oks_mean = oks.mean(axis=0)
    recall_arr = (oks >= oks_thresh).sum(axis=0)/len(oks)

    score_table = pd.DataFrame([oks_mean, recall_arr])
    score_table.columns = label_class
    score_table.index = [OKS_name, "recall"]

    return score_table
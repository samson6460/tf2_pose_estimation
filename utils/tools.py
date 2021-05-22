# Copyright 2020 Samson. All Rights Reserved.
# =============================================================================

"""Utilities and tools for Yolo.
"""
import base64
import json
import os
import threading
from io import BytesIO
from math import ceil, log

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from PIL import Image
import cv2


def read_img(path, size=(512, 512), rescale=None):
    """Read pictures as ndarray.

    Args:
        size: A tuple of 2 integers(heights, widths).
        rescale: A float or None,
            specifying how the image value should be scaled.
            If None, no scaled.
    """
    img_list = [f for f in os.listdir(path) if not f.startswith(".")]
    data = np.empty((len(img_list), *size, 3))

    for i, _path in enumerate(img_list):
        img = Image.open(path + os.sep + _path)
        img = _process_img(img, size)
        data[i] = img
    if rescale is not None:
        data *=rescale
     
    return data


def _process_img(img, size):
    size = size[1], size[0]
    img = img.resize(size)
    img = img.convert("RGB")
    img = np.array(img)
    return img


def read_file(img_path=None,
              label_path=None,
              label_format="labelme",
              img_size=(512, 512),
              label_size="auto",
              label_class=["Head", "Neck", "Hip",
                "L_shoulder", "L_elbow", "L_wrist",
                "R_shoulder", "R_elbow", "R_wrist",
                "L_hip", "L_knee", "L_ankle",
                "R_hip", "R_knee", "R_ankle"],
              sigma=2,
              augmenter=None,
              aug_times=1,
              shuffle=False,
              seed=None,
              thread_num=10):
    """Read the images and annotations created by labelimg or labelme.

    Args:
        img_path: A string, 
            file path of images.
        label_path: A string,
            file path of annotations.
        label_format: A string,
            one of "labelme" and "labelimg(not yet supported)" .
        img_size: A tuple of 2 integer,
            shape of output image(heights, widths).
        label_size: A tuple of 2 integers or a string,
            "auto" means to set size as img_size divided by 4.
            "same" means to set size as img_size.
        label_class: A list of string,
            containing all label names.
        sigma: A integer,
            standard deviation of 2D gaussian distribution.
        augmenter: A `imgaug.augmenters.meta.Sequential` instance.
        aug_times: An integer,
            the default is 1, which means no augmentation.
        shuffle: Boolean, default: True.
        seed: An integer, random seed, default: None.
        thread_num: An integer,
            specifying the number of threads to read files.

    Returns:
        A tuple of 2 ndarrays, (data, label),
        shape of data: (sample_num, img_height, img_width, channel)
        shape of label: (sample_num, label_height, label_width, class_num)
    """
    def _read_labelimg(_path_list, _pos):
        pass

    def _read_labelme(_path_list, _pos):
        for _path_list_i, name in enumerate(_path_list):
            pos = (_pos + _path_list_i)*aug_times

            with open(os.path.join(
                      label_path,
                      name[:name.rfind(".")] + ".json"),
                      encoding="big5") as f:
                jdata = f.read()
                data = json.loads(jdata)

            if img_path is None:
                img64 = data['imageData']
                img = Image.open(BytesIO(base64.b64decode(img64)))
            else:
                img = Image.open(os.path.join(img_path, name))

            zoom_r = np.array(img.size)/np.array(label_size)
            img = _process_img(img, img_size)
            train_data[pos] = img/255.

            for data_i in range(len(data['shapes'])):
                label = data["shapes"][data_i]["label"]
                if label in label_class:
                    index = label_class.index(label)
                    point = np.array(data['shapes'][data_i]['points'])
                    point = point/zoom_r
                    label_im = draw_heatmap(*label_size, point, sigma)
                    label_data[pos][..., index] = label_im
    if label_size == "auto":
        label_size = img_size[0]//4, img_size[1]//4
    elif label_size == "same":
        label_size = img_size
    
    if (label_format == "labelme" 
        and (img_path is None or label_path is None)):
        if label_path is None:
            label_path = img_path
            img_path = None
        path_list = os.listdir(label_path)
        path_list = [f for f in path_list if f.endswith(".json")]
    else:
        path_list = os.listdir(img_path)
        path_list = [f for f in path_list if not f.startswith(".")]
    path_list_len = len(path_list)
    
    class_num = len(label_class)
    if augmenter is None:
        aug_times = 1

    train_data = np.empty((path_list_len*aug_times,
                           *img_size, 3))
    label_data = np.zeros((path_list_len*aug_times,
                           *label_size, class_num))

    threads = []
    workers_num = ceil(path_list_len/thread_num)
    if label_format == "labelimg":
        thread_func = _read_labelimg
    elif label_format == "labelme":
        thread_func = _read_labelme
    else:
        raise ValueError("Invalid format:", label_format)  

    for path_list_i in range(0, path_list_len, workers_num):
        threads.append(
            threading.Thread(target = thread_func,
            args = (
                path_list[path_list_i:path_list_i + workers_num],
                path_list_i))
        )
    for thread in threads:
        thread.start()                
    for thread in threads:
        thread.join()

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        shuffle_index = np.arange(len(train_data))
        np.random.shuffle(shuffle_index)
        train_data = train_data[shuffle_index]
        label_data = label_data[shuffle_index]

    return train_data, label_data


def decode(label, zoom_r):
    class_num = label.shape[-1]
    max_index = label.reshape(-1, class_num).argmax(axis=0)
    x_arr = max_index%label.shape[0]*zoom_r[1]
    y_arr = max_index//label.shape[0]*zoom_r[0]
    return x_arr, y_arr


def vis_img_ann(img, label,
                color=['r', 'lime', 'b', 'c', 'm', 'y',
                    'pink', 'w', 'brown', 'g', 'teal',
                    'navy', 'violet', 'linen', 'gold'],
                connections = [[0, 1, 2],
                            [1, 3, 4, 5],
                            [1, 6, 7, 8],
                            [2, 9, 10, 11],
                            [2, 12, 13, 14]],
                figsize=None,
                dpi=None,
                axis="off",
                savefig_path=None,
                return_fig_ax=False,
                fig_ax=None,
                point_radius=5,
                linewidth=2,
                line_alpha=0.6):
    """Visualize image and annotation.

    Args:
        img: A ndarry of shape(img heights, img widths, color channels).
        label: A ndarray of annotations.
        color: A list of color string or RGB tuple of float.
            Example of color string:
                ['r', 'lime', 'b', 'c', 'm', 'y',
                 'pink', 'w', 'brown', 'g', 'teal',
                 'navy', 'violet', 'linen', 'gold'](Default).
                check for more info about color string by the following url:
                https://matplotlib.org/tutorials/colors/colors.html
            Example of RGB tuple of float:
                [(1, 0, 0), (0, 0, 1)](which means Red、Blue).
        connections: A list of lists of integers.
            The way of key point connection.
            For example, [[0, 2], [1, 3]] means connecting point 0 and point 2,
            and connecting point 1 and point 3.
        figsize: (float, float), optional, default: None
            width, height in inches. If not provided, defaults to [6.4, 4.8].        
        dpi: float, default: rcParams["figure.dpi"] (default: 100.0)
            The resolution of the figure in dots-per-inch.
            Set as 1.325, then 1 inch will be 1 dot.    
        axis: bool or str
            If a bool, turns axis lines and labels on or off.
            If a string, possible values are:
            https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axis.html
        savefig_path: None or string or PathLike or file-like object
            A path, or a Python file-like object.
        return_fig_ax: A boolean.
        fig_ax: (matplotlib.pyplot.figure, matplotlib.pyplot.axes),
            reusing figure and axes can save RAM.
        point_radius: 5.
        linewidth: 2.
        line_alpha: 0.6.
    """
    color = list(map(to_rgb, color))
    class_num = label.shape[-1]
    zoom_r = np.array(img.shape[:2])/np.array(label.shape[:2])

    if fig_ax is None:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    else:
        fig, ax = fig_ax
        ax.clear()
    ax.imshow(img)
    ax.axis(axis)

    x_arr, y_arr = decode(label, zoom_r)

    for class_i in range(class_num):
        x = x_arr[class_i]
        y = y_arr[class_i]
        
        cir = Circle(
            (x, y),
            radius=point_radius,
            color=color[class_i])
        ax.add_patch(cir)

    if connections is not None:
        for connect in connections:
            line = mlines.Line2D(
                x_arr[connect],
                y_arr[connect],
                color=color[connect[0]],
                linewidth=linewidth,
                alpha=line_alpha)
            ax.add_line(line)

    if savefig_path is not None:
        fig.savefig(savefig_path, bbox_inches='tight', pad_inches=0)

    plt.show()

    if return_fig_ax:
        return fig, ax


def draw_img_ann(img, label,
                 color=['r', 'lime', 'b', 'c', 'm', 'y',
                        'pink', 'w', 'brown', 'g', 'teal',
                        'navy', 'violet', 'linen', 'gold'],
                 connections=[[0, 1, 2],
                              [1, 3, 4, 5],
                              [1, 6, 7, 8],
                              [2, 9, 10, 11],
                              [2, 12, 13, 14]],
                 point_radius=8,
                 linewidth=4):
    """Draw image with annotation.

    Args:
        img: A ndarry of shape(img heights, img widths, color channels).
            dtype: uint8, range: [0-255].
        label: A ndarray of annotations.
        color: A list of color string or RGB tuple of float.
            Example of color string:
                ['r', 'lime', 'b', 'c', 'm', 'y',
                 'pink', 'w', 'brown', 'g', 'teal',
                 'navy', 'violet', 'linen', 'gold'](Default).
                check for more info about color string by the following url:
                https://matplotlib.org/tutorials/colors/colors.html
            Example of RGB tuple of float:
                [(1, 0, 0), (0, 0, 1)](which means Red、Blue).
        connections: A list of lists of integers.
            The way of key point connection.
            For example, [[0, 2], [1, 3]] means connecting point 0 and point 2,
            and connecting point 1 and point 3.
        point_radius: 5.
        linewidth: 2.
    """
    color = list(map(to_rgb, color))
    class_num = label.shape[-1]
    zoom_r = np.array(img.shape[:2])/np.array(label.shape[:2])

    x_arr, y_arr = decode(label, zoom_r)
    x_arr = x_arr.astype("int")
    y_arr = y_arr.astype("int")

    for class_i in range(class_num):
        x = x_arr[class_i]
        y = y_arr[class_i]
        c_color = np.array(color[class_i])*255
        img = cv2.circle(img, (x, y),
            point_radius, c_color, -1)

    if connections is not None:
        for connect in connections:
            for point_id in range(len(connect) - 1):
                start_x = x_arr[connect[point_id]]
                start_y = y_arr[connect[point_id]]
                end_x = x_arr[connect[point_id + 1]]
                end_y = y_arr[connect[point_id + 1]]
                l_color = np.array(color[connect[0]])*255
                img = cv2.line(img,
                    (start_x, start_y),
                    (end_x, end_y),
                    l_color, linewidth, cv2.LINE_AA)
    return img


def get_class_weight(label_data, method="alpha"):
    """Get the weight of the category.

    Args:
        label_data: A ndarray of shape(batch_size, grid_num, grid_num, info).
        method: A string,
            one of "alpha"、"log"、"effective"、"binary".

    Returns:
        A list containing the weight of each category.
    """
    class_weight = []
    if method != "alpha":
        total = 1
        for i in label_data.shape[:-1]:
            total *= i
        if method == "effective":
            beta = (total - 1)/total
    for i in range(label_data.shape[-1]):
        samples_per_class = label_data[..., i].sum()
        if method == "effective":
            effective_num = 1 - np.power(beta, samples_per_class)
            class_weight.append((1 - beta)/effective_num)
        elif method == "binary":
            class_weight.append(samples_per_class/(total - samples_per_class))
        else:
            class_weight.append(1/samples_per_class)
    class_weight = np.array(class_weight)
    if method == "log":
        class_weight = np.log(total*class_weight)
 
    if method != "binary":
        class_weight = class_weight/np.sum(class_weight)*len(class_weight)

    return class_weight


def draw_heatmap(height, width,
                 point, sigma, mode="gs"):
    x = np.arange(width, dtype=np.float)
    y = np.arange(height, dtype=np.float)
    if mode == "gs":
        x = (x - point[0])**2
        y = (y - point[1])**2
        xx, yy = np.meshgrid(x, y)
        exp_value = - (xx + yy)/(2*sigma**2)
    elif mode == "exp":
        alpha = (log(2)/2)**0.5/sigma
        x = abs(x - point[0])
        y = abs(y - point[1])
        xx, yy = np.meshgrid(x, y)
        exp_value = - alpha*(xx + yy)
    heatmap = np.exp(exp_value)
    return heatmap
# Copyright 2020 Samson. All Rights Reserved.
# =============================================================================

"""Utilities and tools for pose estimation.
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
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from PIL import Image
import cv2
from imgaug.augmentables import Keypoint, KeypointsOnImage
from tensorflow.keras.utils import Sequence
from tensorflow.python.ops.nn_impl import normalize


def read_img(path, size=(512, 512), rescale=None, preprocessing=None):
    """Read images as ndarray.

    Args:
        path: A string, path of images.
        size: A tuple of 2 integers,
            (heights, widths).
        rescale: A float or None,
            specifying how the image value should be scaled.
            If None, no scaled.
        preprocessing: A function of data preprocessing,
            (e.g. noralization, shape manipulation, etc.)
    """
    img_list = [f for f in os.listdir(path) if not f.startswith(".")]
    data = np.empty((len(img_list), *size, 3))

    for img_i, _path in enumerate(img_list):
        img = Image.open(path + os.sep + _path)
        img = _process_img(img, size)
        data[img_i] = img

    if rescale:
        data = data*rescale
    if preprocessing is not None:
        data = preprocessing(data)
        
    return data


def _process_img(img, size):
    size = size[1], size[0]
    img = img.resize(size)
    img = img.convert("RGB")
    img = np.array(img)
    return img


class Keypoint_reader(object):
    """Read the images and keypoint annotations.

    Args:
        rescale: A float or None,
            specifying how the image value should be scaled.
            If None, no scaled.
        preprocessing: A function of data preprocessing,
            (e.g. noralization, shape manipulation, etc.)
        augmenter: A `imgaug.augmenters.meta.Sequential` instance.
        aug_times: An integer,
            The default is 1, which means no augmentation.

    Attributes:
        rescale
        preprocessing
        augmenter
        aug_times
        file_names: A list of string
            with all file names that have been read.

    Return:
        A reader instance for images and annotations.

    """
    def __init__(self, rescale=None,
                 preprocessing=None,
                 augmenter=None, aug_times=1):
        self.rescale = rescale
        self.preprocessing = preprocessing
        self.augmenter = augmenter
        self.aug_times = aug_times
        self.file_names = None

        if augmenter is None:
            self.aug_times = 1

    def labelme_json_to_dataset(
        self, img_path=None, label_path=None,
        class_names=["Head", "Neck", "Hip",
                     "L_shoulder", "L_elbow", "L_wrist",
                     "R_shoulder", "R_elbow", "R_wrist",
                     "L_hip", "L_knee", "L_ankle",
                     "R_hip", "R_knee", "R_ankle"],
        img_size=(512, 512), label_size="auto",
        heatmap_type="gs", sigma=3.14, normalize=False,
        num_stacks=1, shuffle=False, seed=None,
        encoding="big5", thread_num=10):
        """Read the images and annotations created by labelimg or labelme.

        Args:
            img_path: A string, 
                file path of images.
            label_path: A string,
                file path of annotations.
            class_names: A list of string,
                containing all label names.
            img_size: A tuple of 2 integer,
                shape of output image(heights, widths).
            label_size: A tuple of 2 integers or a string,
                "auto" means to set size as img_size divided by 4.
                "same" means to set size as img_size.
            heatmap_type: A string, one of "gs" or "exp".
                "gs": Gaussian heatmap.
                "exp": Exponential heatmap(proposed by 
                https://pubmed.ncbi.nlm.nih.gov/31683913/).
            sigma: An integer or list of integers.
                standard deviation of 2D gaussian distribution.
            normalize: A boolean,
                whether to normalize each channel in heatmaps
                so that sum would be 1.
            num_stacks: An integer,
                number of stacks of hourglass network.
            shuffle: Boolean, default: True.
            seed: An integer, random seed, default: None.
            thread_num: An integer,
                specifying the number of threads to read files.

        Returns:
            A tuple of 2 ndarrays, (img data, label data),
            a list of tuples like above if num_stacks > 1.
            img data: 
                shape (sample_num, img_height, img_width, channel)
            label data:
                shape (sample_num, label_height, label_width, class_num)
        """
        if hasattr(sigma, '__getitem__'):
            first_sigma = sigma[0]
        else:
            first_sigma = sigma
        img_data, label_data = self._file_to_array(
            img_path=img_path, label_path=label_path,
            class_names=class_names,
            img_size=img_size, label_size=label_size,
            heatmap_type=heatmap_type, sigma=first_sigma,
            normalize=normalize, shuffle=shuffle, seed=seed,
            encoding=encoding,
            thread_num=thread_num, format="labelme")

        if num_stacks > 1:
            if isinstance(sigma, int):
                label_data = [label_data]*num_stacks
            else:
                label_size = label_data.shape[1:3]
                points = heatmap2point(label_data)
                label_data = [label_data]
                for stack_i in range(1, num_stacks):
                    heatmaps = draw_heatmap_batch(
                        label_size, points,
                        sigma=sigma[stack_i],
                        mode=heatmap_type)
                    if normalize:
                        norm = heatmaps.sum(
                            axis=(1, 2), keepdims=True)
                        heatmaps = heatmaps/norm
                    label_data.append(heatmaps)

        return img_data, label_data

    def _process_paths(self, path_list):
        path_list = np.array(path_list)
        U_num = path_list.dtype.itemsize//4 + 5
        dtype = "<U" + str(U_num)
        filepaths = np.empty((len(path_list),
                             self.aug_times),
                             dtype = dtype)
        filepaths[:, 0] = path_list
        filepaths[:, 1:] = np.char.add(filepaths[:, 0:1], "(aug)")
        path_list = filepaths.flatten()
        return path_list
    
    def _file_to_array(self, img_path, label_path,
                       class_names,
                       img_size, label_size,
                       heatmap_type, sigma,
                       normalize,
                       shuffle, seed,
                       encoding,
                       thread_num, format):
        def _encode_to_array(img, kps, pos, indexes):
            for index, keypoint in zip(indexes, kps.keypoints):
                img_data[pos] = img

                point = keypoint.x, keypoint.y
                label_im = draw_heatmap(
                    *label_size, point,
                    sigma, heatmap_type)
                if normalize:
                    norm = label_im.sum(
                        axis=(0, 1), keepdims=True)
                    label_im = label_im/norm
                label_data[pos][..., index] = label_im
        def _imgaug_to_array(img, kps, pos, indexes):
            _encode_to_array(img, kps, pos, indexes)
            if self.augmenter is not None:
                for aug_i in range(1, self.aug_times):
                    img_aug, kps_aug = self.augmenter(
                        image=img, keypoints=kps)
                    _encode_to_array(img_aug, kps_aug,
                        pos + aug_i, indexes)
        def _read_labelimg(_path_list, _pos):
            pass

        def _read_labelme(_path_list, _pos):
            for _path_list_i, name in enumerate(_path_list):
                pos = (_pos + _path_list_i)*self.aug_times

                with open(os.path.join(
                        label_path,
                        name[:name.rfind(".")] + ".json"),
                        encoding=encoding) as f:
                    jdata = f.read()
                    data = json.loads(jdata)

                if img_path is None:
                    img64 = data['imageData']
                    img = Image.open(BytesIO(base64.b64decode(img64)))
                else:
                    img = Image.open(os.path.join(img_path, name))

                zoom_r = (np.array(img.size)
                          /np.array((label_size[1], label_size[0])))
                img = _process_img(img, img_size)

                kps = []
                indexes = []
                for data_i in range(len(data['shapes'])):
                    label = data["shapes"][data_i]["label"]
                    if label in class_names:
                        index = class_names.index(label)
                        indexes.append(index)

                        point = np.array(data['shapes'][data_i]['points'])
                        point = point.squeeze()/zoom_r
                        kps.append(Keypoint(x=point[0],
                                            y=point[1]))
                kps = KeypointsOnImage(kps, shape=label_size)
                _imgaug_to_array(img, kps, pos, indexes)
        if label_size == "auto":
            label_size = img_size[0]//4, img_size[1]//4
        elif label_size == "same":
            label_size = img_size
        
        if (format == "labelme" 
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
        
        class_num = len(class_names)

        img_data = np.empty((path_list_len*self.aug_times,
                            *img_size, 3))
        label_data = np.zeros((path_list_len*self.aug_times,
                            *label_size, class_num))

        threads = []
        workers_num = ceil(path_list_len/thread_num)
        if format == "labelimg":
            thread_func = _read_labelimg
        elif format == "labelme":
            thread_func = _read_labelme
        else:
            raise ValueError("Invalid format:", format)  

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

        if self.rescale is not None:
            img_data = img_data*self.rescale
        if self.preprocessing is not None:
            img_data = self.preprocessing(img_data)

        path_list = self._process_paths(path_list)

        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            shuffle_index = np.arange(len(img_data))
            np.random.shuffle(shuffle_index)
            img_data = img_data[shuffle_index]
            label_data = label_data[shuffle_index]
            path_list = path_list[shuffle_index]

        path_list = path_list.tolist()
        self.file_names = path_list

        return img_data, label_data
    
    def labelme_json_to_sequence(
        self, img_path=None, label_path=None,
        batch_size=20,
        class_names=["Head", "Neck", "Hip",
                     "L_shoulder", "L_elbow", "L_wrist",
                     "R_shoulder", "R_elbow", "R_wrist",
                     "L_hip", "L_knee", "L_ankle",
                     "R_hip", "R_knee", "R_ankle"],
        img_size=(512, 512), label_size="auto",
        heatmap_type="gs", sigma=2,
        normalize=False, num_stacks=1,
        shuffle=False, seed=None,
        encoding="big5", thread_num=1):
        """Convert the JSON file generated by `labelme` into a Sequence.

        Args:
            img_path: A string, 
                file path of images.
            label_path: A string,
                file path of annotations.
            batch_size:  An integer,
                size of the batches of data (default: 20).
            class_names: A list of string,
                containing all label names.
            img_size: A tuple of 2 integer,
                shape of output image(heights, widths).
            label_size: A tuple of 2 integers or a string,
                "auto" means to set size as img_size divided by 4.
                "same" means to set size as img_size.
            heatmap_type: A string, one of "gs" or "exp".
                "gs": Gaussian heatmap.
                "exp": Exponential heatmap(proposed by 
                https://pubmed.ncbi.nlm.nih.gov/31683913/).
            sigma: An integer or list of integers.
                standard deviation of 2D gaussian distribution.
            normalize: A boolean,
                whether to normalize each channel in heatmaps
                so that sum would be 1.
            num_stacks: An integer,
                number of stacks of hourglass network.
            shuffle: Boolean, default: True.
            seed: An integer, random seed, default: None.
            encoding: A string,
                encoding format of file, default: "big5".
            thread_num: An integer,
                specifying the number of threads to read files.

        Returns:
            A tf.Sequence.
                Sequence[i]: A tuple of 2 ndarrays, (img data, label data),
                             a list of tuples like above if num_stacks > 1.
            img data: 
                shape (batches, img_height, img_width, channel)
            label data:
                shape (batches, label_height, label_width, class_num)
        """
        if hasattr(sigma, '__getitem__'):
            first_sigma = sigma[0]
        else:
            first_sigma = sigma
        seq = KeypointSequence(
            img_path=img_path, label_path=label_path,
            batch_size=batch_size, class_names=class_names,
            img_size=img_size, label_size=label_size,
            heatmap_type=heatmap_type, sigma=first_sigma,
            normalize=normalize, label_format="labelme",
            rescale=self.rescale,
            preprocessing=self.preprocessing,
            augmenter=self.augmenter,
            shuffle=shuffle, seed=seed,
            encoding=encoding, thread_num=thread_num)
        self.file_names = seq.path_list

        if num_stacks > 1:
            class StackedKeypointSequence(Sequence):
                def __init__(self, seq, num_stacks):
                    self.seq = seq
                    self.num_stacks = num_stacks
                def __len__(self):
                    return len(self.seq)
                def __getitem__(self, idx):
                    img_data, label_data = self.seq[idx]
                    if isinstance(sigma, int):
                        label_data = [label_data]*num_stacks
                    else:
                        label_size = label_data.shape[1:3]
                        points = heatmap2point(label_data)
                        label_data = [label_data]
                        for stack_i in range(1, num_stacks):
                            heatmaps = draw_heatmap_batch(
                                label_size, points,
                                sigma=sigma[stack_i],
                                mode=heatmap_type)
                            if normalize:
                                norm = heatmaps.sum(
                                    axis=(1, 2), keepdims=True)
                                heatmaps = heatmaps/norm
                            label_data.append(heatmaps)
                    return img_data, label_data
            seq = StackedKeypointSequence(seq, num_stacks)
        return seq


class KeypointSequence(Sequence):

    def __init__(self, img_path=None,
                 label_path=None,
                 batch_size=20,
                 class_names=["Head", "Neck", "Hip",
                              "L_shoulder", "L_elbow", "L_wrist",
                              "R_shoulder", "R_elbow", "R_wrist",
                              "L_hip", "L_knee", "L_ankle",
                              "R_hip", "R_knee", "R_ankle"],
                 img_size=(512, 512),
                 label_size="auto",
                 heatmap_type="gs",
                 sigma=2,
                 normalize=False,
                 label_format="labelme",
                 rescale=1/255,
                 preprocessing=None,
                 augmenter=None,
                 shuffle=False,
                 seed=None,
                 encoding="big5",
                 thread_num=1):
        self.img_path = img_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.class_names = class_names
        self.class_num = len(class_names)
        self.img_size = img_size
        self.heatmap_type = heatmap_type
        self.sigma = sigma     
        self.label_format = label_format
        self.rescale = rescale
        self.preprocessing = preprocessing
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.seed = seed
        self.encoding = encoding
        self.thread_num = thread_num

        if label_size == "auto":
            self.label_size = img_size[0]//4, img_size[1]//4
        elif label_size == "same":
            self.label_size = img_size
        
        if (label_format == "labelme" 
            and (img_path is None or label_path is None)):
            if label_path is None:
                self.label_path = img_path
                self.img_path = None
            path_list = os.listdir(self.label_path)
            self.path_list = [f for f in path_list if f.endswith(".json")]
        else:
            path_list = os.listdir(img_path)
            self.path_list = [f for f in path_list if not f.startswith(".")]

        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            self.path_list = np.array(self.path_list)
            np.random.shuffle(self.path_list)
            self.path_list = self.path_list.tolist()

    def __len__(self):
        return ceil(len(self.path_list)/self.batch_size)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Sequence index out of range")
        def _encode_to_array(img, kps, pos, indexes):
            for index, keypoint in zip(indexes, kps.keypoints):
                img_data[pos] = img

                point = keypoint.x, keypoint.y
                label_im = draw_heatmap(
                    *self.label_size, point,
                    self.sigma, self.heatmap_type)
                if normalize:
                    norm = label_im.sum(
                        axis=(0, 1), keepdims=True)
                    label_im = label_im/norm
                label_data[pos][..., index] = label_im
        
        def _imgaug_to_array(img, kps, pos, indexes):
            if self.augmenter is None:
                _encode_to_array(img, kps, pos, indexes)
            else:
                img_aug, kps_aug = self.augmenter(
                    image=img, keypoints=kps)
                _encode_to_array(img_aug, kps_aug,
                    pos, indexes)

        def _read_labelimg(_path_list, _pos):
            pass

        def _read_labelme(_path_list, _pos):
            for _path_list_i, name in enumerate(_path_list):
                pos = (_pos + _path_list_i)

                with open(os.path.join(
                        self.label_path,
                        name[:name.rfind(".")] + ".json"),
                        encoding=self.encoding) as f:
                    jdata = f.read()
                    data = json.loads(jdata)

                if self.img_path is None:
                    img64 = data['imageData']
                    img = Image.open(BytesIO(base64.b64decode(img64)))
                else:
                    img = Image.open(os.path.join(self.img_path, name))

                zoom_r = (np.array(img.size)
                          /np.array((self.label_size[1], self.label_size[0])))
                img = _process_img(img, self.img_size)

                kps = []
                indexes = []
                for data_i in range(len(data['shapes'])):
                    label = data["shapes"][data_i]["label"]
                    if label in self.class_names:
                        index = self.class_names.index(label)
                        indexes.append(index)

                        point = np.array(data['shapes'][data_i]['points'])
                        point = point.squeeze()/zoom_r
                        kps.append(Keypoint(x=point[0],
                                            y=point[1]))
                kps = KeypointsOnImage(kps, shape=self.label_size)
                _imgaug_to_array(img, kps, pos, indexes)

        total_len = len(self.path_list)
        if (idx + 1)*self.batch_size > total_len:
            batch_size = total_len % self.batch_size
        else:
            batch_size = self.batch_size
        img_data = np.empty((batch_size, *self.img_size, 3))
        label_data = np.zeros((batch_size, *self.label_size,
                               self.class_num))
        start_idx = idx*self.batch_size
        end_idx = (idx + 1)*self.batch_size
        path_list = self.path_list[start_idx:end_idx]
        if self.label_format == "labelimg":
            thread_func = _read_labelimg
        elif self.label_format == "labelme":
            thread_func = _read_labelme
        else:
            raise ValueError("Invalid format: %s" % self.label_format)

        threads = []
        workers = ceil(len(path_list)/self.thread_num)

        for worker_i in range(0, len(path_list), workers):
            threads.append(
                threading.Thread(target=thread_func,
                args=(path_list[worker_i : worker_i+workers],
                      worker_i)))
        for thread in threads:
            thread.start()                
        for thread in threads:
            thread.join()

        if self.rescale is not None:
            img_data = img_data*self.rescale
        if self.preprocessing is not None:
            img_data = self.preprocessing(img_data)
      
        return img_data, label_data


def decode(label, zoom_r, method="mean"):
    height, width = label.shape[:2]
    
    if method == "max":
        num_points = label.shape[-1]
        max_index = label.reshape(-1, num_points).argmax(axis=0)
        
        x_arr = max_index%height*zoom_r[1]
        y_arr = max_index//height*zoom_r[0]
    elif method == "mean":
        label = label/label.sum(axis=(0, 1), keepdims=True)
        
        index_map_x = np.arange(width).reshape((1, -1, 1))
        index_map_y = np.arange(height).reshape((-1, 1, 1))

        label_map_x = label*index_map_x
        label_map_y = label*index_map_y

        x_arr = np.sum(label_map_x, axis=(0, 1))*zoom_r[1]
        y_arr = np.sum(label_map_y, axis=(0, 1))*zoom_r[0]
    
    return x_arr, y_arr


def heatmap2point(heatmaps, zoom_r=(1, 1), method="max"):
    """
    Convert heatmaps to points.

    Args:
        heatmap: An array,
            shape: (batches, heights, widths, num_points).
        zoom_r: An array like of magnification,
            (heights, widths).
        method: One of "max" and "mean".
            "max": Use the brightest position
                of the heatmap as the keypoint.
            "mean": Use the average brightness
                of the heatmap as the keypoint.

    Returns:
        An array, shape: (batches, num_points, 2).
    """
    height, width = heatmaps.shape[1:3]
    
    if method == "max":
        num_points = heatmaps.shape[-1]
        area = height*width
        max_index = heatmaps.reshape(-1, area, num_points).argmax(axis=1)
        
        x_arr = max_index%height*zoom_r[1]
        y_arr = max_index//height*zoom_r[0]
    elif method == "mean":
        heatmaps = heatmaps/heatmaps.sum(axis=(1, 2), keepdims=True)
        
        index_map_x = np.arange(width).reshape((1, 1, -1, 1))
        index_map_y = np.arange(height).reshape((1, -1, 1, 1))

        label_map_x = heatmaps*index_map_x
        label_map_y = heatmaps*index_map_y

        x_arr = np.sum(label_map_x, axis=(1, 2))*zoom_r[1]
        y_arr = np.sum(label_map_y, axis=(1, 2))*zoom_r[0]
    
    points = np.stack((x_arr, y_arr), axis=-1)
    return points


def vis_img_ann(img, label,
                decode_method="max",
                color=['r', 'lime', 'b', 'c', 'm', 'y',
                       'pink', 'w', 'brown', 'g', 'teal',
                       'navy', 'violet', 'linen', 'gold'],
                connections = None,
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
        decode_method: One of "max" and "mean".
            "max": Use the brightest position
                of the heatmap as the keypoint.
            "mean": Use the average brightness
                of the heatmap as the keypoint.
        color: A list of color string or RGB tuple of float.
            Example of color string:
                ['r', 'lime', 'b', 'c', 'm', 'y',
                 'pink', 'w', 'brown', 'g', 'teal',
                 'navy', 'violet', 'linen', 'gold'](Default).
                check for more info about color string by the following url:
                https://matplotlib.org/tutorials/colors/colors.html
            Example of RGB tuple of float:
                [(1, 0, 0), (0, 0, 1)](which means Red、Blue).
        connections: None or a list of lists of integers.
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

    x_arr, y_arr = decode(label, zoom_r, decode_method)

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
                 decode_method="max",
                 color=['r', 'lime', 'b', 'c', 'm', 'y',
                        'pink', 'w', 'brown', 'g', 'teal',
                        'navy', 'violet', 'linen', 'gold'],
                 connections = None,
                 point_radius=8,
                 linewidth=4):
    """Draw image with annotation.

    Args:
        img: A ndarry of shape(img heights, img widths, color channels).
            dtype: uint8, range: [0-255].
        label: A ndarray of annotations.
        decode_method: One of "max" and "mean".
            "max": Use the brightest position
                of the heatmap as the keypoint.
            "mean": Use the average brightness
                of the heatmap as the keypoint.
        color: A list of color string or RGB tuple of float.
            Example of color string:
                ['r', 'lime', 'b', 'c', 'm', 'y',
                 'pink', 'w', 'brown', 'g', 'teal',
                 'navy', 'violet', 'linen', 'gold'](Default).
                check for more info about color string by the following url:
                https://matplotlib.org/tutorials/colors/colors.html
            Example of RGB tuple of float:
                [(1, 0, 0), (0, 0, 1)](which means Red、Blue).
        connections: None or a list of lists of integers.
            The way of key point connection.
            For example, [[0, 2], [1, 3]] means connecting point 0 and point 2,
            and connecting point 1 and point 3.
        point_radius: 5.
        linewidth: 2.
    """
    color = list(map(to_rgb, color))
    class_num = label.shape[-1]
    zoom_r = np.array(img.shape[:2])/np.array(label.shape[:2])

    x_arr, y_arr = decode(label, zoom_r, decode_method)
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


def draw_heatmap_batch(size, point, sigma, mode="gs"):
    batches = point.shape[0]
    class_num = point.shape[-2]
    x = np.arange(size[1], dtype=np.float)
    y = np.arange(size[0], dtype=np.float)
    xx = np.zeros((batches, *size, class_num))
    xx = xx + x.reshape((1, 1, size[1], 1))
    yy = np.zeros((batches, *size, class_num))
    yy = yy + y.reshape((1, size[0], 1, 1))
    if mode == "gs":
        xx = (xx - point[..., 0].reshape(
            batches, 1, 1, class_num))**2
        yy = (yy - point[..., 1].reshape(
            batches, 1, 1, class_num))**2
        exp_value = - (xx + yy)/(2*sigma**2)
    elif mode == "exp":
        alpha = (log(2)/2)**0.5/sigma
        xx = abs(xx - point[..., 0].reshape(
            batches, 1, 1, class_num))
        yy = abs(yy - point[..., 1].reshape(
            batches, 1, 1, class_num))
        exp_value = - alpha*(xx + yy)
    heatmap = np.exp(exp_value)
    return heatmap
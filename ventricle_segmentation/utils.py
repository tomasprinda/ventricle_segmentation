import csv
import errno
import json
import os
import pickle
import shutil

import matplotlib.pylab as pl
# matplotlib.use('Agg')  # prevents _tkinter error
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ventricle_segmentation import cfg
from ventricle_segmentation.core import AnnotatedScan


def iou(mask1, mask2):
    """
    Intersection over union metric
    :param np.ndarrary mask1:
    :param np.ndarrary mask2:
    :return float: Value of the metric
    """
    mask1 = np.ravel(mask1)
    mask2 = np.ravel(mask2)

    i = sum(mask1 & mask2)
    u = sum(mask1 | mask2)

    if u == 0:
        return 0.

    return i / u


def get_epoch_loss(losses_weights):
    """
    Calculates average loss function for epoch
    :param list[(float, int)] losses_weights: List of tuple with loss_values per batch and batch sizes
    :return float: Average loss value
    """
    losses, weights = zip(*losses_weights)
    return np.average(losses, weights=weights)


def plot_annotated_scan(annotated_scan, plot_mask=True):
    """
    :param AnnotatedScan annotated_scan:
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(annotated_scan.dicom_img.astype(float), cmap=pl.cm.hot)
    if plot_mask:
        if annotated_scan.imask is not None:
            plot_contour_mask(annotated_scan.dicom_img, annotated_scan.imask, cmap=pl.cm.Blues)
        if annotated_scan.omask is not None:
            plot_contour_mask(annotated_scan.dicom_img, annotated_scan.omask, cmap=pl.cm.Greens)
    plt.title("\n".join([
        annotated_scan.dicom_file,
        annotated_scan.imask.contours_file if annotated_scan.imask else "no imask",
        annotated_scan.omask.contours_file if annotated_scan.omask else "no omask"
    ]))


def plot_contour_mask(dicom_img, contour_mask, cmap):
    # plt.figure(figsize=(12, 12))
    # plt.imshow(dicom_img.astype(float), cmap=pl.cm.hot)

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    plt.imshow(contour_mask.mask.astype(float) / 4, cmap=my_cmap, alpha=.5)


def prepare_exp_dir(exp_name, clean_dir):
    """
    Prepare folder for experiment
    :param str exp_name:
    :param bool clean_dir:
    """
    cfg.EXP_DIR = os.path.join(cfg.EXPERIMENTS_DIR, exp_name + "/")
    make_sure_path_exists(cfg.EXP_DIR)
    if clean_dir:
        clean_folder(cfg.EXP_DIR)


def clean_folder(folder):
    """
    :param str folder:
    """
    import os
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def backup_files(files, folder):
    for file in files:
        shutil.copy(file, folder)


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def make_sure_path_exists(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def json_load(filename):
    with open(filename, "r") as f:
        return json.load(f)


def json_dump(d, filename):
    with open(filename, "w") as f:
        json.dump(d, f)


def pickle_load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def pickle_dump(d, filename):
    with open(filename, "wb") as f:
        pickle.dump(d, f)


def csv_dump(rows, filename, append=False, delimiter=";"):
    open_param = 'a' if append else 'w'
    with open(filename, open_param) as csvfile:
        writer = csv.writer(csvfile, delimiter=str(delimiter), quotechar=str('"'), quoting=csv.QUOTE_MINIMAL)
        rows = [list(row) for row in rows]  # Like copy, but converts inner tuples to lists
        writer.writerows(rows)


def txt_dump(rows, filename, append=False):
    open_param = 'a' if append else 'w'
    with open(filename, open_param) as txtfile:
        txtfile.writelines(rows)


def strip_extension(filename):
    arr = filename.split(".")
    if len(arr) == 1:
        return arr[0]
    return ".".join(arr[:-1])


def print_info(var, name="var", output=False):
    """
    Print variable information.
    :type var: any
    :type name: str
    :return:
    """
    if isinstance(var, np.ndarray):
        out = "{}: type:{}, shape:{}, dtype:{}, min:{}, max:{}".format(name, type(var), var.shape, var.dtype, np.min(var), np.max(var))
    elif isinstance(var, list) or isinstance(var, tuple):
        out = "{}: type:{}, len:{}, type[0]:{}".format(name, type(var), len(var), type(var[0]) if len(var) > 0 else "")
    else:
        out = "{}: val:{}, type:{}".format(name, var, type(var))
    if output:
        return out
    else:
        print(out)

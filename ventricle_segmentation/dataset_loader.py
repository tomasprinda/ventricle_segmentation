import os

import numpy as np
import torch.utils.data as data

from ventricle_segmentation.core import AnnotatedScan
from ventricle_segmentation.utils import pickle_load, print_info


class ScansDataset(data.Dataset):
    """
    Loader that loads dataset. Dataset is stored in dataset_dir.
    Each example is pickled AnnotatedScan
    See scripts/prepare_data.py to check how to create dataset
    """

    def __init__(self, dataset_dir):
        """
        :param str dataset_dir: Path to folder with pickle files
        """
        self.dataset_dir = dataset_dir

        self.scan_pkl_files = [file for file in os.listdir(self.dataset_dir) if file.endswith(".pkl")]

    def __getitem__(self, index):
        scan_path = os.path.join(self.dataset_dir, self.scan_pkl_files[index])
        annotated_scan = pickle_load(scan_path)  # type: AnnotatedScan

        dicom_img = (annotated_scan.dicom_img.astype(np.float32) - 500) / 1000
        dicom_img = np.expand_dims(dicom_img, 0)  # adding channel at dimension 0, required dims in batch (N,C_in,H_in,W_in); N is not present yet

        imask = annotated_scan.imask.mask.astype(np.int64)

        return dicom_img, imask, annotated_scan.dicom_file, annotated_scan.imask.contours_file

    def __len__(self):
        return len(self.scan_pkl_files)

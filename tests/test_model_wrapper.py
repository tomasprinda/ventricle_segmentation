import unittest

import os
from collections import defaultdict

import torch

from ventricle_segmentation import cfg
from ventricle_segmentation.dataset_loader import ScansDataset
from ventricle_segmentation.model_wrapper import ModelWrapper
from ventricle_segmentation.parsing import parse_dicom_file, np, polygon_to_mask, parse_contour_file, load_all_scans
from ventricle_segmentation.utils import print_info, pickle_load, iou

TRAIN_DIR = os.path.join(cfg.DATASETS_DIR, "train")
DEV_DIR = os.path.join(cfg.DATASETS_DIR, "dev")


class ModelWrapperTestTrainAndSave(ModelWrapper):

    def __init__(self, conf):
        super().__init__(conf)

        self.dicom_files = defaultdict(list)

    def get_model(self):
        pass

    def eval_batch(self, batch, epoch, is_training):
        image, target, dicom_file, icontours_file = batch

        self.dicom_files[(epoch, is_training)] += dicom_file

        return 0, len(dicom_file)

    def log_epoch(self, epoch, train_loss, dev_loss, is_best):
        pass

    def save_checkpoint(self, epoch, best_loss, is_best):
        pass


class TestModelWrapper(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_scan_dataset(self):

        # init
        conf = {
            "batch_size": 8,
            "workers": 2,
            "epochs": 1,
            "learning_rate": 0.001,
        }
        train_dataset = ScansDataset(TRAIN_DIR)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=conf["workers"], pin_memory=True
        )

        # image
        in_channels = 1
        height = 256
        width = 256
        image_shape_expected = [conf["batch_size"], in_channels, height, width]

        # target
        target_shape_expected = [conf["batch_size"], height, width]

        # test loader
        for batch in train_loader:
            image, target, dicom_file, icontours_file = batch

            self.assertListEqual(list(image.shape), image_shape_expected)
            self.assertListEqual(list(target.shape), target_shape_expected)
            break

    def test_train_and_save(self):

        # init
        conf = {
            "batch_size": 8,
            "workers": 2,
            "epochs": 2,
        }
        model_wrapper = ModelWrapperTestTrainAndSave(conf)
        model_wrapper.train_and_save(TRAIN_DIR, DEV_DIR)

        # Should contains the same elements but should be shuffled
        is_training = 1
        l1 = model_wrapper.dicom_files[(0, is_training)]
        l2 = model_wrapper.dicom_files[(1, is_training)]
        self.assertSetEqual(set(l1), set(l2))
        l_equals = [val1 == val2 for val1, val2 in zip(l1, l2)]
        self.assertFalse(all(l_equals))

        # Should be the same order
        is_training = 0
        l1 = model_wrapper.dicom_files[(0, is_training)]
        l2 = model_wrapper.dicom_files[(1, is_training)]
        self.assertListEqual(l1, l2)

    def test_loss(self):

        # Init
        conf = {
            "batch_size": 8,
            "workers": 2,
            "epochs": 1,
            "learning_rate": 0.001,
        }
        model_wrapper = ModelWrapper(conf)
        model_wrapper.get_model()

        train_dataset = ScansDataset(TRAIN_DIR)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=conf["workers"], pin_memory=True
        )

        # test loss value
        epoch = 0
        for i, batch in enumerate(train_loader):

            loss_val, batch_size = model_wrapper.eval_batch(batch, epoch, is_training=True)

            # Loss value should be log(nr_classes) before start training
            loss_val_expected = np.log(2)
            self.assertLess(abs(loss_val - loss_val_expected), 0.1)

            break
        else:
            raise Exception("No data in train_loader")


if __name__ == '__main__':
    unittest.main()

import unittest

import os
from pprint import pprint

import matplotlib
matplotlib.use('Agg')  # prevents _tkinter error
import matplotlib.pyplot as plt

from ventricle_segmentation import cfg
from ventricle_segmentation.core import AnnotatedScan, ContourMask
from ventricle_segmentation.parsing import parse_dicom_file, np, polygon_to_mask, parse_contour_file, load_all_scans, get_contour_mask
from ventricle_segmentation.utils import print_info, pickle_load, iou, prepare_exp_dir, csv_dump, plot_annotated_scan


class TestParsing(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_parse_dicom_file(self):
        dicom_files = [
            os.path.join(cfg.DICOMS_DIR, "SCD0000101", "1.dcm"),
            os.path.join(cfg.DICOMS_DIR, "SCD0000201", "12.dcm"),
            os.path.join(cfg.DICOMS_DIR, "SCD0000301", "200.dcm"),
            os.path.join(cfg.DICOMS_DIR, "SCD0000501", "8.dcm"),
        ]

        for dicom_file in dicom_files:
            dicom_img = parse_dicom_file(dicom_file)
            self.assertIsInstance(dicom_img, np.ndarray)

    def test_get_contour_mask(self):
        test_files = list(os.listdir(cfg.TEST_MASKS_DIR))

        # Test if some test images
        self.assertGreater(len(test_files), 0)

        for test_mask_file in test_files:
            test_mask_file = os.path.join(cfg.TEST_MASKS_DIR, test_mask_file)
            expected_annotated_scan = pickle_load(test_mask_file)  # type: AnnotatedScan

            # Test masks
            masks_dict = {"imask": expected_annotated_scan.imask, "omask": expected_annotated_scan.omask}
            for mask_type, expected_mask in masks_dict.items():
                if expected_mask is not None:
                    shape = expected_annotated_scan.dicom_img.shape
                    imask = get_contour_mask(expected_mask.contours_file, shape)

                    self.assertEqual(imask.mask.tolist(), expected_mask.mask.tolist(), mask_type)
                    self.assertEqual(imask.contours_file, expected_mask.contours_file, mask_type)
                else:
                    self.assertIsNone(expected_mask, mask_type)

        non_existing_path = "/not/existing/file.txt"
        mask = get_contour_mask(non_existing_path, (100, 100))
        self.assertIsNone(mask)

    def test_imask_inside_omask(self):
        prepare_exp_dir("test_imask_inside_omask", clean_dir=True)
        i_wrong = 0

        tolerance = 0.001
        wrong_scores = []
        for annotated_scan in load_all_scans(cfg.LINKS_FILE, cfg.DICOMS_DIR, cfg.CONTOURS_DIR):

            if annotated_scan.imask is None or annotated_scan.omask is None:
                continue  # Skip if some mask not available

            wrong_pixels = annotated_scan.imask.mask > annotated_scan.omask.mask
            wrong_pixels = np.mean(np.ravel(wrong_pixels))

            # Plot incorrect data
            if wrong_pixels > tolerance:
                i_wrong += 1
                row = [i_wrong, wrong_pixels, annotated_scan.dicom_file, annotated_scan.imask.contours_file, annotated_scan.omask.contours_file]
                csv_dump([row], os.path.join(cfg.EXP_DIR, "wrong_files.csv"), append=True)
                plot_annotated_scan(annotated_scan)
                plt.savefig(os.path.join(cfg.EXP_DIR, "{}.png".format(i_wrong)))
                plt.clf()

            wrong_scores.append(wrong_pixels)

        # Test it
        for wrong_pixels in wrong_scores:
            self.assertLess(wrong_pixels, tolerance)

    def test_load_all_scans(self):
        n = 20
        annotated_scans = list(load_all_scans(cfg.LINKS_FILE, cfg.DICOMS_DIR, cfg.CONTOURS_DIR, n=n))

        # Test if some test images
        self.assertEqual(len(annotated_scans), n)

        for annotated_scan in annotated_scans:

            imask_loaded = isinstance(annotated_scan.imask, ContourMask)
            omask_loaded = isinstance(annotated_scan.omask, ContourMask)
            self.assertTrue(imask_loaded or omask_loaded)

            self.assertIsInstance(annotated_scan.dicom_img, np.ndarray)

            if imask_loaded:
                self.assertTupleEqual(annotated_scan.dicom_img.shape, annotated_scan.imask.mask.shape)
                self.assertIsInstance(annotated_scan.imask.contours_file, str)
                self.assertNotEqual(annotated_scan.imask.contours_file, "")
            else:
                self.assertIsNone(annotated_scan.imask)

            if omask_loaded:
                self.assertTupleEqual(annotated_scan.dicom_img.shape, annotated_scan.omask.mask.shape)
                self.assertIsInstance(annotated_scan.omask.contours_file, str)
                self.assertNotEqual(annotated_scan.omask.contours_file, "")
            else:
                self.assertIsNone(annotated_scan.omask)


if __name__ == '__main__':
    unittest.main()

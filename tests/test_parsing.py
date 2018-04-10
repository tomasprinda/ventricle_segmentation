import unittest

import os

from ventricle_segmentation import cfg
from ventricle_segmentation.parsing import parse_dicom_file, np, polygon_to_mask, parse_contour_file, load_all_scans
from ventricle_segmentation.utils import print_info, pickle_load, iou


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

    def test_poly_to_mask(self):
        for test_mask_file in os.listdir(cfg.TEST_MASKS_DIR):
            test_mask_file = os.path.join(cfg.TEST_MASKS_DIR, test_mask_file)
            expected_annotated_scan = pickle_load(test_mask_file)

            expected_mask = expected_annotated_scan.imask

            contours = parse_contour_file(expected_annotated_scan.icontours_file)
            mask = polygon_to_mask(contours, *expected_mask.shape)

            result = iou(mask, expected_mask)
            expected_result = 1
            tolerance = 0.05

            self.assertLess(abs(result - expected_result), tolerance)

    def test_load_all_scans(self):
        annotated_scans = list(load_all_scans(cfg.LINKS_FILE, cfg.DICOMS_DIR, cfg.CONTOURS_DIR, n=20))
        for annotated_scan in annotated_scans:
            self.assertIsInstance(annotated_scan.imask, np.ndarray)
            self.assertIsInstance(annotated_scan.dicom_img, np.ndarray)
            self.assertTupleEqual(annotated_scan.dicom_img.shape, annotated_scan.imask.shape)




if __name__ == '__main__':
    unittest.main()

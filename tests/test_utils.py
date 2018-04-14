import unittest

from ventricle_segmentation.core import AnnotatedScan, ContourMask
from ventricle_segmentation.parsing import np
from ventricle_segmentation.utils import iou, get_pixel_intensities


class TestUtils(unittest.TestCase):

    def __init__(self, methodName='runTest', net_wrapper=None):
        super().__init__(methodName)

    def test_iuo(self):
        maskA = np.array([[True, True], [False, False]])
        maskB = np.array([[True, False], [True, False]])
        maskC = np.array([[False, False], [False, False]])

        tests = [
            ((maskA, maskA), 1.),
            ((maskA, maskB), 1 / 3),
            ((maskA, maskC), 0.),
            ((maskC, maskC), 0.),
        ]

        for (mask1, mask2), expected_result in tests:
            result = iou(mask1, mask2)
            self.assertAlmostEqual(result, expected_result)

    def test_get_pixel_intensities(self):

        dicom_img = np.array([[1,2,3,4], [5,6,7,8]])
        omask = ContourMask(np.array([[True, True, False, False], [True, True, False, False]]), contours_file="")
        imask = ContourMask(np.array([[False, True, False, False], [False, True, False, False]]), contours_file="")
        scan1 = AnnotatedScan(dicom_img, dicom_file="", imask=imask, omask=omask)

        blood_pool_intensities_expected = [2, 6] * 2
        heart_muscle_intensities_expected = [1, 5] * 2
        blood_pool_intensities, heart_muscle_intensities = get_pixel_intensities([scan1] * 2, norm_scan_intenities=False)
        self.assertListEqual(sorted(blood_pool_intensities), sorted(blood_pool_intensities_expected))
        self.assertListEqual(sorted(heart_muscle_intensities), sorted(heart_muscle_intensities_expected))

        blood_pool_intensities_expected = (np.array([2, 6] * 2) - 1) / 5
        heart_muscle_intensities_expected = (np.array([1, 5] * 2) - 1) / 5
        blood_pool_intensities, heart_muscle_intensities = get_pixel_intensities([scan1] * 2, norm_scan_intenities=True)
        self.assertListEqual(sorted(blood_pool_intensities), sorted(blood_pool_intensities_expected))
        self.assertListEqual(sorted(heart_muscle_intensities), sorted(heart_muscle_intensities_expected))


if __name__ == '__main__':
    unittest.main()

import unittest

from ventricle_segmentation.parsing import np
from ventricle_segmentation.utils import iou


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


if __name__ == '__main__':
    unittest.main()

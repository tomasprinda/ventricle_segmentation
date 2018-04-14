class AnnotatedScan:
    """
    Dataset example holder
    """

    def __init__(self, dicom_img, dicom_file, imask, omask):
        """
        :param np.ndarray dicom_img:
        :param str dicom_file: Path to dicom file
        :param ContourMask|None imask: Mask that separates the left ventricular blood pool from the heart muscle (myocardium)
        :param ContourMask|None omask: Mask that defines the outer border of the left ventricular heart muscle
        """
        self.dicom_img = dicom_img
        self.dicom_file = dicom_file
        self.imask = imask
        self.omask = omask


class ContourMask:
    """
    i-contour / o-contour mask holder
    """

    def __init__(self, mask, contours_file):
        """
        :param np.ndarray mask: dtype=bool i/o-contour boolean mask
        :param str contours_file: Path to i/o-countours file (before converting to mask)
        """
        self.mask = mask
        self.contours_file = contours_file

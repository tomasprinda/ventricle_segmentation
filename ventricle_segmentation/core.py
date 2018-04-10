class AnnotatedScan:
    """
    Dataset example holder
    """

    def __init__(self, dicom_img, imask, dicom_file, icontours_file):
        """
        :param np.ndarray dicom_img:
        :param np.ndarray imask: dtype=bool Mask that defines left ventricular blood pool in dicom image
        :param str dicom_file: Path to dicom file
        :param str icontours_file: Path to countours file (before converting to mask)
        """
        self.dicom_img = dicom_img
        self.imask = imask
        self.dicom_file = dicom_file
        self.icontours_file = icontours_file
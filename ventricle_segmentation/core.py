class AnnotatedScan:

    def __init__(self, dicom_img, imask, dicom_file, icontours_file):
        """
        :param np.ndarray dicom_img:
        :param np.ndarray imask:
        :param str dicom_file:
        :param str icontours_file:
        """
        self.dicom_img = dicom_img
        self.imask = imask
        self.dicom_file = dicom_file
        self.icontours_file = icontours_file

        # self.dicom_id = None
        # self.contours_id = None
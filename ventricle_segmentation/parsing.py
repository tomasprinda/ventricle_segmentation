"""Parsing code for DICOMS and contour files"""

import os
import re

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from pydicom.errors import InvalidDicomError

from ventricle_segmentation.core import AnnotatedScan, ContourMask


def load_all_scans(links_file, dicoms_dir, contours_dir, n=None):
    """
    Load all annotated scans for more patients.

    :param str links_file: csv file containing dicom_id-contour_id pairs with header
    :param str contours_dir: dir containing contours files
    :param str dicoms_dir: dir containing dicom files
    :param int|None n: nr of scans to load. Load all if None
    :return list[AnnotatedScan]: iterable
    """
    i = 0
    for (dicom_id, contours_id) in parse_links_file(links_file):

        if dicom_id == "SCD0000501":
            # o-contours failed TestParsing.test_imask_inside_omask().
            # See experiments/test_imask_inside_omask_without_skipping
            continue

        dicom_patients_dir = os.path.join(dicoms_dir, dicom_id)
        icontour_patients_dir = os.path.join(contours_dir, contours_id, "i-contours")
        ocontour_patients_dir = os.path.join(contours_dir, contours_id, "o-contours")

        for annotated_scan in load_patient_scans(dicom_patients_dir, icontour_patients_dir, ocontour_patients_dir):

            if n is not None and i >= n:
                return

            i += 1
            yield annotated_scan


def parse_links_file(filename):
    links_lst = []
    with open(filename, 'r') as infile:
        for i, line in enumerate(infile):

            if i == 0:
                continue  # Skip header

            fields = line.strip().split(",")
            assert len(fields) == 2

            dicom_id, contours_id = fields
            links_lst.append((dicom_id, contours_id))

    return links_lst


pattern_dicom = re.compile("^(\d+)\.dcm")


def load_patient_scans(dicom_patient_dir, icontour_patient_dir, ocontour_patient_dir):
    """
    Loads dicom files with corresponding i-contours and o-contours files if at least one of the contours exists
    for the dicom file.
    Make sure dicom_patient_dir matches icontour_patient_dir (by link.csv)

    :param str dicom_patient_dir: dir containing dicom files {:d}.dcm files
    :param str icontour_patient_dir: dir contating i-contour files IM-0001-{:04d}-icontour-manual.txt
    :param str ocontour_patient_dir: dir contating o-contour files IM-0001-{:04d}-ocontour-manual.txt
    :return list[AnnotatedScan]: iterable
    """

    for dicom_file in os.listdir(dicom_patient_dir):
        match = pattern_dicom.match(dicom_file)

        if match:  # Correct icontour file

            dicom_nr = int(match.group(1))
            dicom_file = os.path.join(dicom_patient_dir, dicom_file)
            icontour_file = os.path.join(icontour_patient_dir, "IM-0001-{:04d}-icontour-manual.txt".format(dicom_nr))
            ocontour_file = os.path.join(ocontour_patient_dir, "IM-0001-{:04d}-ocontour-manual.txt".format(dicom_nr))

            if not os.path.isfile(icontour_file) and not os.path.isfile(ocontour_file):
                continue  # Don't load when no annotation data

            annotated_scan = load_annotated_scan(dicom_file, icontour_file, ocontour_file)
            yield annotated_scan


def load_annotated_scan(dicom_file, icontour_file, ocontour_file):
    """
    Loads AnnotatedScan from dicom and i/o-contours files.
    If i/o-contours files deosn't exists, AnnotatedScan.imask/omask is None
    :param str dicom_file: Path to dicom file
    :param str icontour_file: Path to icontour file (might not exists)
    :param str ocontour_file: Path to icontour file (might not exists)
    :return AnnotatedScan :
    """
    dicom_img = parse_dicom_file(dicom_file)
    imask = get_contour_mask(icontour_file, dicom_img.shape)
    omask = get_contour_mask(ocontour_file, dicom_img.shape)

    return AnnotatedScan(dicom_img, dicom_file, imask, omask)


def get_contour_mask(contour_file, shape):
    """
    Loads contours file and converts it to mask
    :param str contour_file: Contours file to load
    :param (int, int) shape: Size of mask to create == size of a dicom file
    :return ContourMask|None: Returns None if file doesn't exists
    """
    if not os.path.isfile(contour_file):
        return None

    contours = parse_contour_file(contour_file)
    mask = polygon_to_mask(contours, *shape)
    return ContourMask(mask, contour_file)


def parse_contour_file(filename):
    """
    Parse the given contour filename

    :param str filename: filepath to the contourfile to parse
    :return list[(float, float)]: list of tuples holding x, y coordinates of the contour
    """

    coords_lst = []

    with open(filename, 'r') as infile:
        for line in infile:
            coords = line.strip().split()

            x_coord = float(coords[0])
            y_coord = float(coords[1])
            coords_lst.append((x_coord, y_coord))

    return coords_lst


def parse_dicom_file(filename):
    """
    Parse the given DICOM filename

    :param str filename: filepath to the DICOM file to parse
    :return np.ndarray: DICOM image data dtype=int16
    """

    try:
        dcm = pydicom.read_file(filename)
        dcm_image = dcm.pixel_array

        try:
            intercept = dcm.RescaleIntercept
        except AttributeError:
            intercept = 0.0
        try:
            slope = dcm.RescaleSlope
        except AttributeError:
            slope = 0.0

        if intercept != 0.0 and slope != 0.0:
            dcm_image = dcm_image * slope + intercept

        return dcm_image

    except InvalidDicomError:
        return None


def polygon_to_mask(polygon, width, height):
    """
    Convert polygon to mask

    :param list[(float, float)] polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
     in units of pixels
    :param int width: scalar image width
    :param int height: scalar image height
    :return np.ndarray: Boolean mask of shape (height, width), dtype=bool
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask

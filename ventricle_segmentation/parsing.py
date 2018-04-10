"""Parsing code for DICOMS and contour files"""

import os
import re

import numpy as np
import pydicom
from PIL import Image, ImageDraw
from pydicom.errors import InvalidDicomError

from ventricle_segmentation.core import AnnotatedScan


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

        dicom_patients_dir = os.path.join(dicoms_dir, dicom_id)
        contour_patients_dir = os.path.join(contours_dir, contours_id, "i-contours")

        for annotated_scan in load_patient_scans(dicom_patients_dir, contour_patients_dir):

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


def load_patient_scans(dicom_patient_dir, icontour_patient_dir):
    """
    Loads contours from icontour_patient_dir and loads also corresponding dicom images for annotations from dicom_patient_dir.
    Make sure dicom_patient_dir matches icontour_patient_dir (by link.csv)

    :param str dicom_patient_dir: dir containing dicom files {:d}.dcm files
    :param icontour_patient_dir: dir contating contour files IM-0001-{:04d}-icontour-manual.txt
    :return list[AnnotatedScan]: iterable
    """
    pattern = re.compile("^IM-0001-(\d+)-icontour-manual\.txt")
    for icontour_file in os.listdir(icontour_patient_dir):
        match = pattern.match(icontour_file)
        icontour_file = os.path.join(icontour_patient_dir, icontour_file)

        if match:  # Correct icontour file
            dicom_nr = int(match.group(1))
            dicom_file = os.path.join(dicom_patient_dir, "{}.dcm".format(dicom_nr))
            dicom_img = parse_dicom_file(dicom_file)

            contours = parse_contour_file(icontour_file)
            imask = polygon_to_mask(contours, *dicom_img.shape)

            annotated_scan = AnnotatedScan(dicom_img, imask, dicom_file, icontour_file)
            yield annotated_scan


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


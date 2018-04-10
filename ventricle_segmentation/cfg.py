import os

DATA_DIR = "/www/data/prinda/ventricle_segmentation/final_data/"
PROJECT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

LINKS_FILE = os.path.join(DATA_DIR, "link.csv")
DICOMS_DIR = os.path.join(DATA_DIR, "dicoms")
CONTOURS_DIR = os.path.join(DATA_DIR, "contourfiles")

TEST_MASKS_DIR = os.path.join(DATA_DIR, "test_masks")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, "experiments")
EXP_DIR = os.path.join(EXPERIMENTS_DIR, "exp")

CONF_DEFAULT = os.path.join(PROJECT_DIR, "conf", "conf_default.json")


import os

import click

from ventricle_segmentation import cfg
from ventricle_segmentation.parsing import load_all_scans
from ventricle_segmentation.utils import make_sure_path_exists, backup_files, pickle_dump


@click.command()
@click.option('--datasets_dir', default=cfg.DATASETS_DIR, help='Ouput dir where prepared datasets with examples will be stored.')
@click.option('--train_ratio', default=0.8)
@click.option('--links_file', default=cfg.LINKS_FILE, help='csv file containing dicom_id-contour_id pairs with header')
@click.option('--dicoms_dir', default=cfg.DICOMS_DIR, help='dir containing dicom files')
@click.option('--contours_dir', default=cfg.CONTOURS_DIR, help='dir containing contours files')
def main(datasets_dir, train_ratio, links_file, dicoms_dir, contours_dir):

    # Clean
    if os.path.exists(datasets_dir):
        raise Exception("{} already exists. Choose different of delete it manually.".format(datasets_dir))

    train_dir = os.path.join(datasets_dir, "train/")
    dev_dir = os.path.join(datasets_dir, "dev/")
    make_sure_path_exists(train_dir)
    make_sure_path_exists(dev_dir)
    backup_files([__file__], datasets_dir)  # To see how the data was created

    # Split
    all_scans = list(load_all_scans(links_file, dicoms_dir, contours_dir))
    split = int(len(all_scans) * train_ratio)
    train, dev = all_scans[:split], all_scans[split:]

    # Store
    i_example = 1
    for dataset, dataset_dir in [(train, train_dir), (dev, dev_dir)]:
        for scan in dataset:
            filename = os.path.join(dataset_dir, "{:04d}.pkl".format(i_example))
            pickle_dump(scan, filename)
            i_example += 1


if __name__ == '__main__':
    main()

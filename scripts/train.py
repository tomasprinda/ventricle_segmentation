import click
import os

from ventricle_segmentation import cfg
from ventricle_segmentation.model_wrapper import ModelWrapper
from ventricle_segmentation.utils import prepare_exp_dir, json_load, json_dump, backup_files


@click.command()
@click.option('--conf_file', default=cfg.CONF_DEFAULT, help='Model configuration file')
@click.option('--exp', default="exp", help='Experiment folder name')
# @click.option('--short_run/--long_run', default=False, help='Run on short/long data')
@click.option('--train_dir', required=True, help='Path to train dir.')
@click.option('--dev_dir', required=True, help='Path to dev dir.')
def main(conf_file, exp, train_dir, dev_dir):
    # Clean
    prepare_exp_dir(exp, clean_dir=True)
    backup_files(
        [
            __file__,
            os.path.join(cfg.PROJECT_DIR, "ventricle_segmentation/dataset_loader.py"),
            os.path.join(cfg.PROJECT_DIR, "ventricle_segmentation/model_wrapper.py"),
            os.path.join(cfg.PROJECT_DIR, "ventricle_segmentation/external.py")
        ]
        , cfg.EXP_DIR)

    # Conf
    conf = json_load(conf_file)
    json_dump(conf, cfg.EXP_DIR + "conf.json")

    # Model
    model_wrapper = ModelWrapper(conf)
    model_wrapper.train_and_save(train_dir, dev_dir)


if __name__ == '__main__':
    main()

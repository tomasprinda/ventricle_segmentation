import click

from ventricle_segmentation import cfg
from ventricle_segmentation.utils import backup_files, pickle_load, json_load, prepare_exp_dir, get_class


@click.command()
@click.option('--exp', default="exp", help='Experiment folder name with a model.')
@click.option('--dataset_dir', required=True, help='Path to dataset to predict.')
def main(exp, dataset_dir):

    # Clean
    prepare_exp_dir(exp, clean_dir=False)
    backup_files([__file__], cfg.EXP_DIR)

    # Model
    conf = json_load(cfg.EXP_DIR + "conf.json")
    # ModelWrapper = get_class(cfg.EXP_DIR + "model_wrapper.py", "ModelWrapper")
    from ventricle_segmentation.model_wrapper import ModelWrapper
    model_wrapper = ModelWrapper(conf)
    model_wrapper.predict(dataset_dir)


if __name__ == '__main__':
    main()

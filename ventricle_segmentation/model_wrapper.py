import os
import shutil

import torch
from torch.backends import cudnn

from ventricle_segmentation import cfg, external
from ventricle_segmentation.dataset_loader import ScansDataset
from ventricle_segmentation.utils import get_epoch_loss, txt_dump


class ModelWrapper:
    """
    Wrapper for model that ensures training is done with storing best models, logging results during training for each epoch,
    """

    def __init__(self, conf):
        """
        :param dict[str, Any] conf: Configuration file. Should contain keys ["learning_rate", "workers", "batch_size", "epochs"]
        """
        self.conf = conf

        self.model = None
        self.optimizer = None
        self.criterion = None

    def get_model(self):
        """
        Defines used model, loss function and optimizer
        """
        self.model = external.FCN8s(in_channels=1, n_class=2).cuda()

        # define loss function (criterion) and optimizer
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.conf["learning_rate"])

        cudnn.benchmark = True

    def train_and_save(self, train_dir, dev_dir):
        """
        Trains model, evaluates after each epoch, loggs loss function in each epoch, stores checkpoints and best results
        :param str train_dir: Dir with pickled training examples. See scripts/prepare_data.py to create it.
        :param str dev_dir: Dir with pickled dev examples. See scripts/prepare_data.py to create it.
        """
        self.get_model()

        # Dataset loader
        train_dataset = ScansDataset(train_dir)
        dev_dataset = ScansDataset(dev_dir)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.conf["batch_size"], shuffle=True, num_workers=self.conf["workers"], pin_memory=True
        )
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=self.conf["batch_size"], shuffle=False, num_workers=self.conf["workers"], pin_memory=True
        )

        best_loss = 1e10

        for epoch in range(self.conf["epochs"]):
            train_losses = []
            dev_losses = []

            # train
            for i, batch in enumerate(train_loader):
                loss_val, batch_size, _ = self.eval_batch(batch, epoch, is_training=True)
                train_losses.append((loss_val, batch_size))
            train_loss = get_epoch_loss(train_losses)

            # eval
            for i, batch in enumerate(dev_loader):
                loss_val, batch_size, _ = self.eval_batch(batch, epoch, is_training=False)
                dev_losses.append((loss_val, batch_size))
            dev_loss = get_epoch_loss(dev_losses)

            # handle best
            is_best = dev_loss < best_loss
            best_loss = min(dev_loss, best_loss)
            self.save_checkpoint(epoch, best_loss, is_best)

            # log
            self.log_epoch(epoch, train_loss, dev_loss, is_best)

    def predict(self, dataset_dir):
        self.get_model()
        self.load_checkpoint(cfg.EXP_DIR + "best.pth.tar")

        # Dataset loader
        dataset = ScansDataset(dataset_dir)
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.conf["batch_size"], shuffle=False, num_workers=self.conf["workers"], pin_memory=True
        )

        for i, batch in enumerate(dataset_loader):
            _, _, prediction = self.eval_batch(batch, epoch=None, is_training=False)



    def eval_batch(self, batch, epoch, is_training):
        """
        Runs forward pass, calculates loss function and if is_training runs backward_pass.
        :param tuple batch: Tuple of (image, target, dicom_file, icontours_file). The same as is returned by ScansDataset().__getitem__() with
                added batch dimension at axis=0
        :param int epoch: Epoch nr
        :param bool is_training:
        :return (float, int): Tuple of (loss, batch_size)
        """
        image, target, dicom_file, icontours_file = batch

        # wrap tensors
        image = image.cuda()
        target = target.cuda()
        image_var = torch.autograd.Variable(image)
        target_var = torch.autograd.Variable(target)

        prediction = self.model(image_var)
        loss = self.criterion(prediction, target_var)

        if is_training:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # return
        loss_val = loss.data.cpu().numpy()[0]
        batch_size = len(dicom_file)
        print(prediction)
        prediction = prediction > 0.5
        return loss_val, batch_size, prediction

    def log_epoch(self, epoch, train_loss, dev_loss, is_best):
        """
        Logging of learning proccess after each batch to csv file
        :param int epoch:
        :param float train_loss:
        :param float dev_loss:
        :param bool is_best:
        """

        star = "*" if is_best else ""
        rows = ["{:d},{:.2f},{:2f}{}\n".format(epoch, train_loss, dev_loss, star)]

        if epoch == 0:
            header = ["Epoch,train loss,dev loss\n"]
            rows = header + rows

        txt_dump(rows, cfg.EXP_DIR + "training_log.csv", append=True)

    def save_checkpoint(self, epoch, best_loss, is_best):
        """
        Stores current checkpoint to EXP_DIR and best model if is_best
        :param int epoch:
        :param float best_loss:
        :param bool is_best:
        """

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_loss': best_loss,
            'optimizer': self.optimizer.state_dict(),
        }

        checkpoint_filename = os.path.join(cfg.EXP_DIR, "checkpoint.pth.tar")
        best_filename = os.path.join(cfg.EXP_DIR, "best.pth.tar")

        torch.save(state, checkpoint_filename)

        if is_best:
            shutil.copyfile(checkpoint_filename, best_filename)

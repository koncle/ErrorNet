import os
from pathlib import Path
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch

from framework.DataSet import Image2ImageTrainDataSet
from modules.UNet import UNet
from framework.loss import dice_loss, SidedBCELoss, BCELoss2D
from framework.logger import *
from framework.file_utils import *

from dataset.transforms import resizeN, random_horizontal_flipN
from framework.visualize import show_graph


class TrainFrameWork(object):
    def __init__(self, dataset, data_loader, net=None, optimizer=None,
                 loss_func=BCELoss2D(), evaluate_func=dice_loss,
                 epoch=20, load_model_path=None, save_model_path="./",
                 seed=23202, cuda=True):
        """
        :param dataset:  dataset is the set used to get data_loader
        :param data_loader:      is a dataloader provided by pytorch
        :param net:              is your network
        :param optimizer:        is your optimizer
        :param loss_func:        is your criterion
        :param epoch:            is your trianing epoch
        :param load_model_path:  is where you load your model
        :param save_model_path:  is where to save your model
        :param seed:             is the random seed
        :param cuda:             whether use cuda
        """
        # basic
        self._save_path = save_model_path
        self._epoch_num = epoch
        self._start_epoch = 0
        self._cuda = cuda
        self._very_very_detail = None

        # Random Setting
        self.set_seed(seed)
        self._dataset = dataset
        self._dataloader = data_loader

        # net
        self._optimizer = optimizer
        self._net = net
        self._best_net = None
        self._best_optimizer = None

        # loss
        self._evaluate = evaluate_func
        self._loss_func = loss_func

        if self._net is not None and cuda:
            self._net = self._net.cuda()

        if load_model_path is not None:
            self.load_model(load_model_path)

        # Log hooks
        self._hooks = []
        self.register_hook(ConsolePrintLogger())

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def save_model(self, path):
        self._save_model(path, 1000)

    def _save_model(self, path, epoch):
        # print("Saved model to " + path)
        path = Path(path)
        torch.save({
            'state_dict': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'epoch': epoch,
        }, path / ('%03d.pth' % epoch))

    def load_model(self, path):
        checkpoint = torch.load(path)
        self._start_epoch = checkpoint['epoch']
        self._net.load_state_dict(checkpoint['state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])

        print("Loaded model from " + path)

        if self._start_epoch == 1000:
            return

        if self._start_epoch >= self._epoch_num:
            raise Exception("Wrong epoch_num for current loaded model, "
                            "epoch_num : %d, current start_epoch  %d"
                            % (self._epoch_num, self._start_epoch))

    def inference(self, input, label):
        input = Variable(input)
        label = Variable(label)
        if self._cuda:
            input = input.cuda()
            label = label.cuda()

        logits = self._net(input)

        # logger
        self._inference_hook(input, label, logits)

        # get accuracy with dice_loss
        acc = self._evaluate(logits, label)
        # calculate loss by user
        loss = self._loss_func(logits, label)

        return acc, loss, logits

    def _train_net(self, dataloader):
        """
        Train current net with dataloader
        :param dataloader:
        :return:
        """
        # prepare for calculating loss and accuracy
        self._net.train()
        self._dataset.train_mode()

        sum_loss = 0
        sum_accuracy = 0
        num = 0
        total_step = len(dataloader)
        for step, (batch_x, batch_y) in enumerate(dataloader):
            # forward
            acc, loss, _ = self.inference(batch_x, batch_y)
            # reset grad
            self._optimizer.zero_grad()
            # backward
            loss.backward()
            # update
            self._optimizer.step()
            # calculate total loss and accuracy
            num += 1
            sum_loss += loss.item()
            sum_accuracy += acc.item()

            # output current state
            self._train_step_log_hook(step, total_step, loss, acc)

        average_loss = sum_loss / num
        average_accuracy = sum_accuracy / num
        return average_loss, average_accuracy

    def _validate_net(self, dataloader):
        """
        Validate current model
        :param dataloader:  a  dataloader, it should be set to run in validate mode
        :return:
        """
        self._net.eval()
        sum_loss = 0
        sum_accuracy = 0
        num = 0
        for step, (batch_x, batch_y) in enumerate(dataloader):
            acc, loss, logits = self.inference(batch_x, batch_y)
            num += 1
            sum_loss += loss.item()
            sum_accuracy += acc.item()
            self._validate_hook(batch_x, batch_y, logits)

        average_loss = sum_loss / num
        average_accuracy = sum_accuracy / num
        return average_loss, average_accuracy

    def train(self, output_dir=None):
        if output_dir is None:
            output_dir = self._save_path

        if self._cuda:
            self._net.cuda()

        best_accuracy = 0

        # create folder to save checkpoints
        output_dir = Path(output_dir)
        output_model_dir = output_dir / 'checkpoint'
        best_model_dir = output_dir / 'best'
        create_dir_if_not_exist(output_dir)
        create_dir_if_not_exist(output_model_dir)
        create_dir_if_not_exist(best_model_dir)

        # start train log
        self._start_train_hook()

        for epoch in range(self._start_epoch, self._epoch_num):
            # TODO : reduce couple between dataset and dataloader
            # Train train_set
            self._dataset.train_mode()
            average_loss, average_accuracy = self._train_net(self._dataloader)

            # validate set
            self._dataset.validate_mode()
            val_loss, val_acc = self._validate_net(self._dataloader)

            # output info
            self._train_epoch_log_hook(epoch, self._epoch_num, average_loss, average_accuracy, val_loss, val_acc)

            # save model
            self._save_model(output_model_dir, epoch)

            if best_accuracy < val_acc:
                best_accuracy = val_acc
                self._save_model(best_model_dir, 1000)

        # test set
        self._dataset.test_mode()
        if len(self._dataloader) > 0:
            val_loss, val_acc = self._validate_net(self._dataloader)

        self._end_train_hook(average_loss, average_accuracy, val_loss, val_acc)

    def predict(self, imgs):
        imgs = Variable(imgs)
        if self._cuda:
            imgs = imgs.cuda()
        logits = self._net(imgs)
        return torch.round(torch.sigmoid(logits)).detach().cpu().numpy()

    """        A Clean Hook Implementation              """
    def register_hook(self, hook):
        self._hooks.append(hook)

    def unregister_hook(self, hook):
        if hook in self._hooks:
            self._hooks.remove(hook)

    def _train_step_log_hook(self, iter, num_iter, loss, acc):
        for hook in self._hooks:
            if hasattr(hook, "train_step_hook"):
                hook.train_step_hook(iter, num_iter, loss, acc)

    def _train_epoch_log_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        for hook in self._hooks:
            if hasattr(hook, "train_epoch_hook"):
                hook.train_epoch_hook(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc)

    def _start_train_hook(self):
        for hook in self._hooks:
            if hasattr(hook, "start_train_hook"):
                hook.start_train_hook()

    def _end_train_hook(self, average_loss, average_accuracy, val_loss, val_acc):
        for hook in self._hooks:
            if hasattr(hook, "end_train_hook"):
                hook.end_train_hook(average_loss, average_accuracy, val_loss, val_acc)

    def _inference_hook(self, input, label, logits):
        for hook in self._hooks:
            if hasattr(hook, "inference_hook"):
                hook.inference_hook(input, label, logits)

    def _validate_hook(self, input, label, logits):
        for hook in self._hooks:
            if hasattr(hook, "validate_hook"):
                hook.validate_hook(input, label, logits)


def test_framework():
    # global resized_shape, train_augment, dataset
    check_out_path = '/data/zj/pycharmProjects/ErrorNet/framework/checkpoint/018.pth'
    train_path = '/data/zj/data/18_3_6/'
    batch_size = 4
    num_epoches = 10

    in_shape = (1, 256, 256)
    is_bn = True

    def train_augment(img, label):
        resized_shape = (256, 256)
        img, label = resizeN([img, label], resized_shape)
        img, label = random_horizontal_flipN([img, label])
        return img, label

    # prepare data
    dataset = Image2ImageTrainDataSet(train_path, transforms=[lambda x, y: train_augment(x, y)], mask_suffix="_1.bmp")
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True,
                            num_workers=6, pin_memory=False)
    # prepare net
    net = UNet(in_shape, is_bn)
    optimizer = Adam(lr=1e-4, params=net.parameters(), weight_decay=0.005)
    criterion = SidedBCELoss(weight=[1, 1], pos_weight=None)
    # set framework
    frame_work = TrainFrameWork(dataset, dataloader,
                                net, optimizer, criterion,
                                epoch=num_epoches, show_graph=True,
                                load_model_path=check_out_path)
    # train
    frame_work.train()


if __name__ == '__main__':
    test_framework()

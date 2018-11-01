import os
from pathlib import Path
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch

from framework.ImageDataSet import ImageDataSet
from modules.UNet import UNet
from framework.loss import dice_loss, SidedBCELoss, BCELoss2D

from dataset.transforms import resizeN, random_horizontal_flipN
from framework.visualize import show_graph


class TrainFrameWork(object):
    def __init__(self, dataset, data_loader, net, optimizer, criterion=BCELoss2D(),
                 epoch=20, checkpoint_path=None, model_save_path="./",
                 seed=23202, cuda=True, show_graph=False):
        # basic
        self._save_path = model_save_path
        self._epoch_num = epoch
        self._start_epoch = 0
        self._cuda = cuda
        self._show_graph = show_graph
        self._very_very_detail = None

        # Random Setting
        self.set_seed(seed)
        self._dataset = dataset
        self._dataloader = data_loader

        # net
        self._optimizer = optimizer
        self._net = net

        # loss
        self._evaluate = dice_loss
        self._criterion = criterion

        if cuda:
            self._net = self._net.cuda()

        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

        # Log hooks
        self._step_hooks = []
        self._epoch_hooks = []

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def save_model(self, path, epoch):
        torch.save(self._net.state_dict(), path + '/snap/%03d.pth' % epoch)
        torch.save({
            'state_dict': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'epoch': epoch,
        }, path + '/checkpoint/%03d.pth' % epoch)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self._start_epoch = checkpoint['epoch']
        self._net.load_state_dict(checkpoint['state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded model from " + path)
        if self._start_epoch < self._epoch_num:
            raise  Exception("Wrong epoch_num for current loaded model, "
                             "epoch_num : %d, current start_epoch  %d"
                             % (self._epoch_num, self._start_epoch))

    def inference(self, imgs, masks):
        imgs = Variable(imgs)
        masks = Variable(masks)
        if self._cuda:
            imgs = imgs.cuda()
            masks = masks.cuda()
        logits = self._net(imgs)
        pred = torch.round(torch.sigmoid(logits))

        # show img, prediction, mask with a small probability
        # in every procedure
        if self._very_very_detail is not None:
            very_detail_rate = 0.1
            if np.random.rand() < very_detail_rate:
                show_graph(imgs[0, 0, ...], pred.cpu().detach().numpy()[0, 0, ...], masks[0, 0, ...])

        # get accuracy with dice_loss
        acc = self._evaluate(pred, masks)
        # calculate loss by user
        loss = self._criterion(logits, masks)
        return acc, loss, logits

    def _train_net(self, dataloader):
        """
        Train current net with dataloader
        :param dataloader:
        :return:
        """
        # prepare for calculating loss and accuracy
        sum_loss = 0
        sum_accuracy = 0
        num = 0
        num_iter = len(dataloader)
        for iter, (imgs, masks) in enumerate(dataloader):
            # forward
            acc, loss, _ = self.inference(imgs, masks)
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
            print('\r[{}/{}]       {:.4f}/{:.4f}'
                  .format(iter, num_iter, loss, acc), end='', flush=True)
            self._train_step_log_hook(iter, num_iter, loss, acc)
        average_loss = sum_loss / num
        average_accruracy = sum_accuracy / num
        return average_loss, average_accruracy

    def _validate_net(self, dataloader):
        """
        Validate current model
        :param dataloader:  a  dataloader, it should be set to run in validate mode
        :return:
        """
        sum_loss = 0
        sum_accuracy = 0
        num = 0
        show = True
        for step, (imgs, masks) in enumerate(dataloader):
            acc, loss, logits = self.inference(imgs, masks)

            if self._show_graph and show:
                pred = torch.round(torch.sigmoid(logits))
                show_graph(imgs[0, 0, ...], pred.cpu().detach().numpy()[0, 0, ...], masks[0, 0, ...])
                # only show once in a validation process
                show = False

            num += 1
            sum_loss += loss.item()
            sum_accuracy += acc.item()

        average_loss = sum_loss / num
        average_accuracy = sum_accuracy / num
        return average_loss, average_accuracy

    def train(self, output_dir=None):
        if output_dir is None:
            output_dir = self._save_path

        for epoch in range(self._start_epoch, self._start_epoch + self._epoch_num):
            print('Epoch      train_loss/ acc      val_loss/acc')

            # TODO : reduce couple between dataset and dataloader
            # Train train_set
            self._net.train()
            self._dataset.train_mode()
            average_loss, average_accuracy = self._train_net(self._dataloader)

            # validate set
            self._net.eval()
            self._dataset.validate_mode()
            val_loss, val_acc = self._validate_net(self._dataloader)

            # output info
            print('\r[{}/{}]     {:.4f}/{:.4f}     {:.4f}/{:.4f}'
                  .format(epoch, self._epoch_num, average_loss, average_accuracy, val_loss, val_acc), flush=False)
            self._train_epoch_log_hook(epoch, self._epoch_num, average_loss, average_accuracy, val_loss, val_acc)

            # save model
            self.save_model(output_dir, epoch)

        # test set
        self._net.eval()
        self._dataset.test_mode()
        val_loss, val_acc = self._validate_net(self._dataloader)
        print('\rTest      {:.4f}/{:.4f}     {:.4f}/{:.4f}\n'
              .format(epoch, self._epoch_num, average_loss, average_accuracy, val_loss, val_acc))
        self.save_model(output_dir, 1000)

    def predict(self, imgs):
        imgs = Variable(imgs)
        if self._cuda:
            imgs = imgs.cuda()
        logits = self._net(imgs)
        return torch.round(torch.sigmoid(logits)).detach().cpu().numpy()

    def set_config(self, config):
        self._config = config

    def start_config(self):
        if self._config is None:
            raise Exception("You should set config first!!!")

        if "criterion" in self._config.keys():
            dic = self._config["criterion"]
            func = dic["func"]
            for num, v in enumerate(dic["value"]):
                print("Trying parameter " + "criterion " + str(v))
                self._criterion = func(v)

                output_path = Path(self._save_path + "/" + "criterion_" + str(num))
                if not output_path.exists():
                    os.mkdir(str(output_path))
                self.train()

    """   Log Hooks   """
    def register_log_train_step_hook(self, hook):
        self._step_hooks.append(hook)

    def unregister_log_train_step_hook(self, hook):
        if hook in self._step_hooks:
            self._step_hooks.remove(hook)

    def _train_step_log_hook(self, iter, num_iter, loss, acc):
        for hook in self._step_hooks:
            hook(iter, num_iter, loss, acc)

    def register_log_train_epoch_hook(self, hook):
        self._epoch_hooks.append(hook)

    def unregister_log_train_epoch_hook(self, hook):
        if hook in self._epoch_hooks:
            self._epoch_hooks.remove(hook)

    def _train_epoch_log_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        for hook in self._epoch_hooks:
            hook(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc)



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
    dataset = ImageDataSet(train_path, transforms=[lambda x, y: train_augment(x, y)], mask_suffix="_1.bmp")
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
                                checkpoint_path=check_out_path)
    # train
    frame_work.train()


if __name__ == '__main__':
    test_framework()

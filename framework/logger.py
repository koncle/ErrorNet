import torch
import numpy as np

from framework.visualize import show_graph


class BaseLogger(object):
    def train_epoch_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        pass

    def train_step_hook(self, it, num_iter, loss, acc):
        pass

    def inference_hook(self, input, label, logits):
        pass

    def validate_hook(self, input, label, logits):
        pass

    def start_train_hook(self):
        pass

    def end_train_hook(self, average_loss, average_accuracy, val_loss, val_acc):
        pass


class ConsolePrintLogger(BaseLogger):
    def start_train_hook(self):
        print('Epoch      train_loss/ acc      val_loss/acc')

    def train_epoch_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        print('\r[{}/{}]     {:.4f}/{:.4f}     {:.4f}/{:.4f}'
              .format(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc), flush=False)

    def train_step_hook(self, it, num_iter, loss, acc):
        print('\r[{}/{}]       {:.4f}/{:.4f}'.format(it, num_iter, loss, acc), end='', flush=True)

    def end_train_hook(self, average_loss, average_accuracy, val_loss, val_acc):
        print('\rTest      {:.4f}/{:.4f}     {:.4f}/{:.4f}\n'
              .format(average_loss, average_accuracy, val_loss, val_acc))


class FileLogger(BaseLogger):
    def __init__(self, file_name, framework):
        self._file_name = file_name
        self._framework = framework

    def train_epoch_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        s = '\r[{}/{}]     {:.4f}/{:.4f}     {:.4f}/{:.4f}'.format(epoch, total_epoch, average_loss, average_accuracy,
                                                                   val_loss, val_acc)
        self._log.write(s)

    def start_train_hook(self):
        self._log.write('\nNew train...')

    def end_train_hook(self, average_loss, average_accuracy, val_loss, val_acc):
        self._log.write("\n")

    def close(self):
        self._log.close()

    def __enter__(self):
        self._framework.register_hook(self)
        self._log = open(self._file_name, "w+")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._framework.unregister_hook(self)
        self._log.close()


class ImageShowLogger(BaseLogger):
    def __init__(self):
        self.show = True

    def train_epoch_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        self.show = True

    def validate_hook(self, input, label, logits):
        if self.show:
            pred = torch.round(torch.sigmoid(logits))
            show_graph(input[0, 0, ...], pred.cpu().detach().numpy()[0, 0, ...], label[0, 0, ...])
            # only show once in a validation process
        self.show = False

    def inference_hook(self, input, label, logits):
        # pred = torch.round(torch.sigmoid(logits))
        # show_graph(input[3, 0, ...], pred.cpu().detach().numpy()[3, 0, ...], label[3, 0, ...])
        pass

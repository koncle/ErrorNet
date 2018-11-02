# Hooks for output
def print_epoch_hook(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
    print('\r[{}/{}]     {:.4f}/{:.4f}     {:.4f}/{:.4f}'
          .format(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc), flush=False)


def print_step_hook(it, num_iter, loss, acc):
    print('\r[{}/{}]       {:.4f}/{:.4f}'.format(it, num_iter, loss, acc), end='', flush=True)


def print_start_train_hook():
    print('Epoch      train_loss/ acc      val_loss/acc')


def print_end_train_hook(average_loss, average_accuracy, val_loss, val_acc):
    print('\rTest      {:.4f}/{:.4f}     {:.4f}/{:.4f}\n'
          .format(average_loss, average_accuracy, val_loss, val_acc))


def get_file_logger(file_path):
    log = open(file_path, "w+")

    # register hooks
    def epoch_hook(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        nonlocal log
        s = '\r[{}/{}]     {:.4f}/{:.4f}     {:.4f}/{:.4f}'.format(epoch, total_epoch, average_loss, average_accuracy,
                                                                   val_loss, val_acc)
        log.write(s)

    return epoch_hook


class BaseLogger(object):
    def train_epoch_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        pass

    def train_step_hook(self, it, num_iter, loss, acc):
        pass

    def start_train_hook(self):
        pass

    def end_train_hook(self, average_loss, average_accuracy, val_loss, val_acc):
        pass


class ConsolePrintLogger(BaseLogger):
    def train_epoch_hook(self, epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        print('\r[{}/{}]     {:.4f}/{:.4f}     {:.4f}/{:.4f}'
              .format(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc), flush=False)

    def train_step_hook(self, it, num_iter, loss, acc):
        print('\r[{}/{}]       {:.4f}/{:.4f}'.format(it, num_iter, loss, acc), end='', flush=True)

    def start_train_hook(self):
        print('Epoch      train_loss/ acc      val_loss/acc')

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



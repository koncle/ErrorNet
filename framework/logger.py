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

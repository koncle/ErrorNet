from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.transforms import resizeN, random_horizontal_flipN
from framework.ImageDataSet import ImageDataSet
from framework.loss import SidedBCELoss
from framework.trainframe import TrainFrameWork
from modules.UNet import UNet


def test_framework():
    # global resized_shape, train_augment, dataset
    check_out_path = None  # '/data/zj/pycharmProjects/ErrorNet/framework/checkpoint/018.pth'
    train_path = '/data/zj/data/18_3_6/'
    output_path = '/data/zj/data/output'
    batch_size = 4
    num_epoches = 15

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
                                checkpoint_path=check_out_path,
                                model_save_path=output_path)

    log_num = 0
    log = open(output_path + "/log" + str(log_num) + ".txt", "w+")
    # register hooks
    def epoch_hook(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc):
        nonlocal log, log_num
        s = '\r[{}/{}]     {:.4f}/{:.4f}     {:.4f}/{:.4f}\n' \
                  .format(epoch, total_epoch, average_loss, average_accuracy, val_loss, val_acc)
        log.write(s)
    frame_work.register_log_train_epoch_hook(epoch_hook)

    def step_hook(it, num_iter, loss, acc):
        nonlocal log
        s = '\r[{}/{}]       {:.4f}/{:.4f}'.format(it, num_iter, loss, acc)
        log.write(s)
    frame_work.register_log_train_step_hook(step_hook)

    config = {
        "criterion": {
            "value": [[0.1, 1], [0.3, 1], [0.6, 1], [0.9, 1], [1.2, 1], [1.5, 1], [2, 1]],
            "func": lambda weight: SidedBCELoss(weight=weight, pos_weight=None)
        },
    }
    frame_work.set_config(config)
    frame_work.start_config()
    # train
    # frame_work.train()


if __name__ == '__main__':
    test_framework()

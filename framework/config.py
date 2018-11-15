import torch
from pathlib import Path
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.transforms import resizeN, random_horizontal_flipN
from framework.DataSet import Image2ImageTrainDataSet, ImageTestDataSet
from framework.file_utils import copy_file
from framework.logger import FileLogger, ImageShowLogger
from framework.loss import SidedBCELoss
from framework.trainframe import TrainFrameWork, create_dir_if_not_exist
from framework.visualize import show_output_imgs
from modules import FlexibleUnet
from modules.UNet import UNet
from skimage.io import imread, imsave
import itertools


def get_parametrised_model():
    batch_size = 4
    num_epoches = 30
    weight_list = [[1.2, 1], [1.5, 1], [2, 1], [4, 1], [6, 1]]
    for weight in weight_list:
        net = UNet((1, 256, 256), True)
        optimizer = Adam(lr=1e-4, params=net.parameters(), weight_decay=0.005)
        loss = SidedBCELoss(weight=weight, pos_weight=None)
        yield net, optimizer, loss


class ParameterConfig(object):
    def __init__(self, train_framework, config, output_path):
        self._config = config
        self._output_path = output_path
        self._train_framework = train_framework
        self._init_config()

    def set_config(self, config):
        self._config = config

    def _init_config(self):
        if self._config is None:
            raise Exception("You should set a new config first!!!")

        if "net" not in self._config.keys():
            raise Exception("You need to specify a new network for every model")

        self.new_net_func = self._config["net"]["func"]
        self.func_list = []

        keys = ["criterion", "opt", "epoch_num"]
        value_list = []
        for key in keys:
            if key in self._config.keys():
                dic = self._config[key]
                value_list.append(dic["value"])
                self.func_list.append(dic["func"])
        self.combinations = list(itertools.product(*value_list))

    def config_parameter(self, value=None):
        if value is None:
            value = self.combinations[0]
        self._train_framework._net = self.new_net_func()
        self._train_framework._criterion = self.func_list[0](value[0])
        self._train_framework._optimizer = self.func_list[1](value[1], self._train_framework._net)
        self._train_framework._epoch_num = self.func_list[2](value[2])
        self._train_framework._start_epoch = 0

    def start_parameter_selecting(self, final_model_output_path):
        for value in self.combinations:
            self.config_parameter(value)

            print("Trying parameter " + str(value))
            output_path = Path(self._output_path + "/param_" + str(value))
            create_dir_if_not_exist(output_path)
            with FileLogger(str(output_path / 'log.txt'), self._train_framework):
                self._train_framework.train(str(output_path))
                copy_file(str(output_path / 'best' / '1000.pth'),
                          final_model_output_path + '/param_' + str(value) + ".pth")

    def test_imgs_and_save_result_with_different_models(self, models_path, dataset, output_path):
        model_path_list = list(Path(models_path).iterdir())
        output_path = Path(output_path)
        for model_path in model_path_list:
            self._train_framework.load_model(str(model_path))
            dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=3)
            for step, (img, file_name) in enumerate(dataloader):
                predict = self._train_framework.predict(img)
                b = predict.shape[-1]
                a = predict.shape[-2]
                predict = np.reshape(predict, (a, b))
                imsave(str(output_path / (file_name[0] + "_" + model_path.stem + ".bmp")), predict)

    def test_model_and_save_result(self, model_path, dataset, output_path):
        model_path = Path(model_path)
        self.config_parameter()
        output_path = Path(output_path)
        self._train_framework.load_model(str(model_path))
        dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=3)
        for step, (img, file_name) in enumerate(dataloader):
            predict = self._train_framework.predict(img)
            b = predict.shape[-1]
            a = predict.shape[-2]
            predict = np.reshape(predict, (a, b))
            imsave(str(output_path / (file_name[0] + "_" + model_path.stem + ".bmp")), predict)


def test_framework():
    # global resized_shape, train_augment, dataset
    check_out_path = None  # '/data/zj/data/output/checkpoint/014.pth'
    train_path = '/data/zj/data/kidney/train'
    test_path = '/data/zj/data/kidney/test'
    output_path = '/data/zj/data/output'

    batch_size = 4
    num_epoches = 30

    def train_augment(img, label):
        resized_shape = (256, 256)
        img, label = resizeN([img, label], resized_shape)
        img, label = random_horizontal_flipN([img, label])
        return img, label

    # prepare data
    dataset = Image2ImageTrainDataSet(path=None,
                                      train_set_path=train_path,
                                      test_set_path=test_path,
                                      transforms=[lambda x, y: train_augment(x, y)], mask_suffix="_1.bmp")
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True,
                            num_workers=6, pin_memory=False)

    # set framework
    frame_work = TrainFrameWork(dataset, dataloader,
                                epoch=num_epoches,
                                load_model_path=check_out_path,
                                save_model_path=output_path)
    frame_work.register_hook(ImageShowLogger())
    return frame_work


def config_train(frame_work):
    # [0.005, 1], [0.01, 1],
    # [[0.1, 1], [0.3, 1], [0.6, 1],
    #                       [0.9, 1], [1.2, 1], [1.5, 1], [2, 1], [4, 1], [6, 1]],
    #
    config = {
        "net": {
            "func": lambda: FlexibleUnet.UNet(1),
        },
        "opt": {
            "value": [1e-4],
            "func": lambda v, net: Adam(lr=1e-4, params=net.parameters(), weight_decay=0.005),
        },
        "criterion": {
            "value": [10, 100],
            "func": lambda weight: SidedBCELoss(weight=None, pos_weight=None, area_weight=weight)
        },
        "epoch_num": {
            "value": [20],
            "func": lambda v: v
        }
    }
    output_path = '/data/zj/data/output'
    models_path = '/data/zj/data/models'
    parameterConfig = ParameterConfig(train_framework=frame_work, config=config, output_path=output_path)
    parameterConfig.start_parameter_selecting(models_path)

    small_test_path = '/data/zj/data/small_test_1'
    dataset = ImageTestDataSet(path=small_test_path)
    parameterConfig.test_imgs_and_save_result_with_different_models(models_path, dataset, small_test_path)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    framework = test_framework()
    config_train(framework)
    show_output_imgs()

import torch
from pathlib import Path
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.transforms import resizeN, random_horizontal_flipN
from framework.ImageTrainDataSet import ImageTrainDataSet, ImageTestDataSet
from framework.logger import FileLogger
from framework.loss import SidedBCELoss
from framework.trainframe import TrainFrameWork, create_dir_if_not_exist
from framework.visualize import show_output_imgs
from modules.UNet import UNet

from skimage.io import imread, imsave


class ParameterConfig(object):
    def __init__(self, train_framework, config, output_path):
        self._config = config
        self._output_path = output_path
        self._train_framework = train_framework

    def set_config(self, config):
        self._config = config

    def start_parameter_selecting(self, final_model_output_path):
        if self._config is None:
            raise Exception("You should set a new config first!!!")

        if "net" not in self._config.keys():
            raise Exception("You need to specify a new network for every model")

        new_net_func = self._config["net"]
        new_opt_func = self._config["opt"]

        if "criterion" in self._config.keys():
            dic = self._config["criterion"]
            func = dic["func"]
            for num, v in enumerate(dic["value"]):
                print("Trying parameter " + "criterion " + str(v))
                self._train_framework.criterion = func(v)

                output_path = Path(self._output_path + "/" + "criterion_" + str(v))
                create_dir_if_not_exist(output_path)
                create_dir_if_not_exist(output_path / 'checkpoint')

                with FileLogger(str(output_path / 'log.txt'), self._train_framework):
                    self._train_framework._net = new_net_func()
                    self._train_framework._optimizer = new_opt_func(self._train_framework._net)
                    self._train_framework.train(str(output_path))
                    self._train_framework.save_model(final_model_output_path + '/' + "criterion_" + str(v) +".pth")


    def test_imgs_and_save_result(self, model, dataset, output_path):
        dataset.train_mode()
        dataset.set_need_file_name(True)
        output_path = Path(output_path)
        self._train_framework.load_model(model)
        dataloader = DataLoader(dataset, batch_size=1, drop_last=False)
        for step, (img, mask, file_name) in enumerate(dataloader):
            predict = self._train_framework.predict(img)
            b = predict.shape[-1]
            a = predict.shape[-2]
            predict = np.reshape(predict, (a, b))
            imsave(str(output_path / (file_name[0] + ".jpg")), predict)

    def test_imgs_and_save_result_with_different_models(self, models_path, dataset, output_path):
        model_path_list = list(Path(models_path).iterdir())
        output_path = Path(output_path)
        for model_path in model_path_list:
            self._train_framework.load_model(str(model_path))
            dataloader = DataLoader(dataset, batch_size=1, drop_last=False)
            for step, (img, file_name) in enumerate(dataloader):
                predict = self._train_framework.predict(img)
                b = predict.shape[-1]
                a = predict.shape[-2]
                predict = np.reshape(predict, (a, b))
                imsave(str(output_path / (file_name[0] + model_path.stem +".bmp")), predict)

def test_framework():
    # global resized_shape, train_augment, dataset
    check_out_path = None  # '/data/zj/data/output/checkpoint/014.pth'
    train_path = '/data/zj/data/18_3_6/train'
    test_path = '/data/zj/data/18_3_6/test'
    output_path = '/data/zj/data/output'
    batch_size = 4
    num_epoches = 30

    in_shape = (1, 256, 256)
    is_bn = True

    def train_augment(img, label):
        resized_shape = (256, 256)
        img, label = resizeN([img, label], resized_shape)
        img, label = random_horizontal_flipN([img, label])
        return img, label

    # prepare data
    dataset = ImageTrainDataSet(path=None,
                                train_set_path=train_path,
                                test_set_path=test_path,
                                transforms=[lambda x, y: train_augment(x, y)], mask_suffix="_1.bmp")
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
    # framework.train()
    return frame_work

def config_train(frame_work):
    # [0.005, 1],[0.01, 1], [0.1, 1], [0.3, 1], [0.6, 1], [0.9, 1],
    config = {
        "net": lambda: UNet((1, 256, 256), True).cuda(),
        "opt": lambda net: Adam(lr=1e-4, params=net.parameters(), weight_decay=0.005),
        "criterion": {
            "value": [[1.2, 1], [1.5, 1], [2, 1], [4, 1], [6, 1]],
            "func": lambda weight: SidedBCELoss(weight=weight, pos_weight=None)
        },
        "learning_rate": {
            "value": [1e-3, 1e-4, 1e-5],
            "func": lambda net, v: Adam(lr=v - 4, params=net.parameters(), weight_decay=0.005),
        },
    }
    output_path = '/data/zj/data/output'
    models_path = '/data/zj/data/models'
    parameterConfig = ParameterConfig(train_framework=frame_work, config=config, output_path=output_path)
    # parameterConfig.start_parameter_selecting(models_path)

    small_test_path = '/data/zj/data/small_test_1'
    test_dataset = ImageTestDataSet(path=small_test_path)
    parameterConfig.test_imgs_and_save_result_with_different_models(models_path, test_dataset, small_test_path)



if __name__ == '__main__':
    framework = test_framework()
    config_train(framework)
    show_output_imgs()

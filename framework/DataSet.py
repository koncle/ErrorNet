import torch
import numpy as np
from glob import glob
from skimage.io import imread, imsave
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
from dataset.transforms import *
from pathlib import Path

class ImageTestDataSet(Dataset):
    def __init__(self, path,
                 img_suffix=".jpg"):
        super(ImageTestDataSet, self).__init__()
        self._img_suffix = img_suffix
        self._test_set = self._load_img_file_name(path)

    def _load_img_file_name(self, path):
        """
        Return path/**/*.img_suffix
        """
        # Search all files in path with suffix is _img_suffix
        return np.array(glob(str(Path(path) / ("**/*" + self._img_suffix)), recursive=True))

    def _format_img(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        if img.shape[-1] == 3:
            img = img.swapaxes(0, 2)
            img = img.swapaxes(1, 2)
        return img

    def _get_item(self, index):
        # get their file names
        img_file_name = self._test_set[index]
        img = imread(img_file_name, as_gray=True).astype(np.float32)
        img = self._format_img(img)
        intensity_min = np.min(img)
        intensity_max = np.max(img)
        img = (img - intensity_min) / (intensity_max - intensity_min)
        # Convert to tensor
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img, Path(img_file_name).stem

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self._test_set)


class SimpleImageDataSet(Dataset):
    def __init__(self, path, img_suffix=".jpg", transforms=None,
                 seed=22222):
        super(SimpleImageDataSet, self).__init__()
        self._img_suffix = img_suffix
        self._transforms = transforms

        self._data, self._dev_set, self._test_set = None, None, None

        np.random.seed(seed)
        # Try optimize this area
        if path is not None:
            self._data = self._load_img_file_name(Path(path))
            np.random.shuffle(self._data)
        else:
            raise Exception("None path specified!!")
        print(self)

    def _load_img_file_name(self, path):
        """
        Return path/**/*.img_suffix
        """
        # Search all files in path with suffix is _img_suffix
        return np.array(glob(str(path / ("**/*" + self._img_suffix)), recursive=True))

    def _format_img(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        if img.shape[-1] == 3:
            img = img.swapaxes(0, 2)
            img = img.swapaxes(1, 2)
        return img

    def _get_item(self, dataset, index):
        # get their file names
        img_file_name = dataset[index]
        # load img
        img = imread(img_file_name, as_gray=True).astype(np.float32)
        if self._transforms is not None:
            for t in self._transforms:
                img = t(img)

        img = self._format_img(img)

        # Convert to tensor
        img = torch.from_numpy(img).type(torch.FloatTensor)
        return img

    def __getitem__(self, index):
        return self._get_item(self._data, index)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return "data set : {}".format(len(self._data))


class Image2ImageTrainDataSet(Dataset):
    def __init__(self, path, mode='train',
                 img_suffix=".jpg", mask_suffix=".bmp", transforms=None,
                 train_ration=0.8, test_ration=0.2, k_fold=None,
                 train_set_path=None, test_set_path=None,
                 seed=22222):
        super(Image2ImageTrainDataSet, self).__init__()
        self._img_suffix = img_suffix
        self._mask_suffix = mask_suffix
        self._transforms = transforms
        self._mode = mode
        self._k_fold = k_fold
        self._need_file_name=False

        self._train_set, self._dev_set, self._test_set = None, None, None

        # Try optimize this area
        if train_set_path is not None:
            self._train_set = self._load_img_file_name(Path(train_set_path))
            if test_set_path is not None:
                self._dev_set = self._load_img_file_name(Path(test_set_path))
        else:
            if path is None:
                raise Exception("No path found to split file!!")
            self._path = Path(path)
            # Split train test set from a whole file
            # load image file_name list(not contain mask)
            self._img_file_names = self._load_img_file_name(self._path)

            if k_fold is None:
                self._train_set, self._dev_set, _ = self._split_files(train_ration, test_ration, k_fold, seed)
            else:
                # split file into a test set, and a generator to generate train set and dev set
                self._train_and_dev_set, self._test_set, self._train_dev_set_idx_generator = self._split_files(
                                                                                                    train_ration,
                                                                                                    test_ration,
                                                                                                    k_fold, seed)
            # generate next train set and dev set
            # self.next_fold()
        print(self)

    def _load_img_file_name(self, path):
        """
        Return path/**/*.img_suffix
        """
        # Search all files in path with suffix is _img_suffix
        return np.array(glob(str(path / ("**/*" + self._img_suffix)), recursive=True))

    def _split_files(self, train_ration, test_ration, k_fold, seed):
        """
        Split total set into [train_set, dev_set, test_set],
           train_set and dev_set are used for training model
           test_set is used for testing model when model is trained.
        """
        assert (train_ration + test_ration == 1)
        # Split total set to train_and_dev set and test set
        # get index of two set
        train_and_dev_index, test_index = train_test_split(list(range(len(self._img_file_names))),
                                                           train_size=train_ration,
                                                           test_size=test_ration,
                                                           random_state=seed)

        # get file name list of two set
        test_set = self._img_file_names[test_index]
        train_and_dev_set = self._img_file_names[train_and_dev_index]
        if k_fold is None:
            return train_and_dev_set, test_set, None
        else:
            # Split this big set into train set and dev set, with k fold validation
            train_dev_set_generator = KFold(k_fold, shuffle=True, random_state=seed).split(train_and_dev_set)
            return train_and_dev_set, test_set, train_dev_set_generator

    def next_fold(self):
        """
        You have to call it yourself to generate new train data and dev data
        :return:
        """
        # generate next index of the train set and dev set
        train_set_idx, dev_set_idx = next(self._train_dev_set_idx_generator)
        # update train set and dev set
        self._train_set = self._train_and_dev_set[train_set_idx]
        self._dev_set = self._train_and_dev_set[dev_set_idx]

    def _get_img_and_label(self, img_file_name):
        mask_file_name = img_file_name.replace(self._img_suffix, self._mask_suffix)

        # load img
        img = imread(img_file_name, as_gray=True).astype(np.float32)

        # load mask
        label = imread(mask_file_name, as_gray=True).astype(np.float32)

        return img, label

    def _format_img(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        if img.shape[-1] == 3:
            img = img.swapaxes(0, 2)
            img = img.swapaxes(1, 2)
        return img

    def _format_img_and_label(self, img, label):
        img = self._format_img(img)
        label = self._format_img(label)
        return img, label

    def _transform_img_and_label(self, img, label):
        """
        Transform img and label simultaneously with specified transforms.
        :param img:  ndarray
        :param label:  ndarray
        :return:
        """
        if self._transforms is not None:
            for t in self._transforms:
                img, label = t(img, label)
        return img, label

    def _normalize_img_and_label(self, img, label):
        """
        TODO: more nomalization operations
        Normalize img and label to fasten learning
        :param img:
        :param label:
        :return:
        """
        intensity_min = np.min(img)
        intensity_max = np.max(img)
        img = (img - intensity_min) / (intensity_max - intensity_min)

        label = label / 255
        return img, label

    def set_need_file_name(self, need_file_name):
        self._need_file_name = need_file_name

    def _get_item(self, dataset, index):
        if dataset is None:
            raise Exception("Error! Do you forget to call next_fold() to generate new data, "
                            "or use dev_mode() to run in no dev data environment?")

        # get their file names
        img_file_name = dataset[index]
        img, label = self._get_img_and_label(img_file_name)
        img, label = self._transform_img_and_label(img, label)
        img, label = self._format_img_and_label(img, label)
        img, label = self._normalize_img_and_label(img, label)

        # Convert to tensor
        img = torch.from_numpy(img).type(torch.FloatTensor)
        # Convert to [0 or 1]
        label = torch.from_numpy((label > 0).astype(np.float32)).type(torch.FloatTensor)
        if self._need_file_name:
            return img, label, Path(img_file_name).stem
        else:
            return img, label

    def validate_mode(self):
        self._mode = 'val'

    def train_mode(self):
        self._mode = 'train'

    def test_mode(self):
        self._mode = 'test'

    def __getitem__(self, index):
        if self._mode == 'train':
            return self._get_item(self._train_set, index)
        elif self._mode == 'val':
            return self._get_item(self._dev_set, index)
        elif self._mode == 'test':
            return self._get_item(self._test_set, index)
        else:
            raise Exception("Unknown mode")

    def __len__(self):
        if self._mode == 'train':
            return len(self._train_set)
        elif self._mode == 'val':
            return len(self._dev_set)
        elif self._mode == 'test':
            return 0 if self._test_set is None else len(self._test_set)
        else:
            raise Exception("Unknown mode")

    def __repr__(self):
        return "train set : {}, dev set : {}, test set : {}".format(len(self._train_set),
                                                                    len(self._dev_set)
                                                                    if self._dev_set is not None else "no",
                                                                    len(self._test_set)
                                                                    if self._test_set is not None else "no")


def check_dataset():
    def train_augment(img, label):
        # img, label = resizeN([img, label], (512, 512))
        img = img.reshape((512, 512))
        label = label.reshape((512, 512))
        img, label = random_horizontal_flipN([img, label])
        return img, label

    def test_augment(img, label):
        img, label = resizeN([img, label], (512, 512))
        return img, label

    test_data = Image2ImageTrainDataSet('/data/zj/data/ct_kidney/train_yuan/', transforms=[lambda x, y: train_augment(x, y), ],
                                        mode='train')
    loader = DataLoader(test_data, batch_size=1, num_workers=0)

    for it, (img, label) in enumerate(loader):
        img = tensor_to_image(img, mean=0, std=1)
        label = tensor_to_label(label)
        imsave('img.png', img.reshape((512, 512)))
        imsave('label.png', label.reshape((512, 512)))


def tensor_to_image(tensor, mean=0, std=1):
    image = tensor.numpy()
    image = image * std + mean
    image = image * 255
    image = image.astype(dtype=np.uint8)
    return image


def tensor_to_label(tensor):
    label = tensor.numpy() * 255
    label = label.astype(dtype=np.uint8)
    return label


if __name__ == '__main__':
    check_dataset()

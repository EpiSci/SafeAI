# Copyright (c) 2018 Episys Science, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from safeai.datasets import mnist
from safeai.datasets import fashion_mnist
from safeai.datasets import cifar10
from safeai.datasets import cifar100
from safeai.datasets import stl10
from safeai.datasets import svhn
from safeai.datasets import tinyimagenet200

IMAGE_DATASET_MAP = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'stl10': stl10,
    'svhn': svhn,
    'tinyimagenet200': tinyimagenet200,
    'mnist': mnist,
    'fashion_mnist': fashion_mnist,
}

class DatasetFetcher:
    def fetch(self, dataset_name):
        raise NotImplementedError


class ImageDatasetFetcher(DatasetFetcher):

    def __init__(self, shape=(32, 32)):
        if shape:
            assert isinstance(shape, (list, tuple))
            assert len(shape) == 2
            self.shape = shape


    def fetch(self, dataset_name):
        self.exist_check(dataset_name)
        return IMAGE_DATASET_MAP[dataset_name].load_data(self.shape)


    def fetch_train_generator(self, dataset_names):
        for dataset_name in dataset_names:
            yield single_element(dataset_name, 'train')


    def fetch_test_generator(self, dataset_names):
        for dataset_name in dataset_names:
            yield single_element(dataset_name, 'test')


    def single_element(self, dataset_name, test_or_train='train'):
        if test_or_train == 'test':
            _, (x_data, y_data) = self.fetch(dataset_name)
        else:
            (x_data, y_data), _ = self.fetch(dataset_name)
        for x, y in zip(x_data, y_data):
            yield x, y


    def exist_check(self, dataset_name):
        dataset_name = dataset_name.lower()
        if dataset_name not in IMAGE_DATASET_MAP.keys():
            raise ValueError("Dataset not supported: specify one of {}"
                             .format(IMAGE_DATASET_MAP.keys()))

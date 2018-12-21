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

RGB_DATA_MAP = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'stl10': stl10,
    'svhn': svhn,
    'tinyimagenet200': tinyimagenet200,
}

GRAY_DATA_MAP = {
    'mnist': mnist,
    'fashion_mnist': fashion_mnist,
}

class DatasetFetcher:
    def __init__(self):
        pass



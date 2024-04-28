import numpy as np
import sys
sys.path.append("/Users/chizhaosheng/Documents/Cources/CV/HW1")
from fashion_mnist.utils import mnist_reader
'''导入MNIST数据集'''
X_train, y_train = mnist_reader.load_mnist('../fashion_mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../fashion_mnist/data/fashion', kind='t10k')
def load_dataset():
    train_image = X_train[:50000]
    train_label = y_train[:50000]
    val_image = X_train[50000:]
    val_label = y_train[50000:]
    test_image = X_test
    test_label = y_test
    return (train_image, train_label, val_image, val_label, test_image, test_label)

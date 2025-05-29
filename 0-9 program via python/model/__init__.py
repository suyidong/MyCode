# models/__init__.py
from .CNN import CnnNet  # 显式导出CNN类
from .mnist_dataset import MnistData  # 显式导出数据集类

__all__ = ['CnnNet', 'MnistData']  # 声明可公开访问的模块
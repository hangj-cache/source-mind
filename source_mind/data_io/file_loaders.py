import scipy.io
import numpy as np
import os
from typing import Union


def load_mat_matrix(file_path: str, var_name: str = None) -> Union[np.ndarray, dict]:
    """
    加载 .mat 文件中的数据。

    :param file_path: .mat 文件的完整路径。
    :param var_name: 要加载的特定变量名。如果为 None，则返回包含所有变量的字典。
    :return: 变量数据（通常为 NumPy 数组或字典）。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")

    try:
        mat_contents = scipy.io.loadmat(file_path)

        # 移除 loadmat 默认添加的元数据键
        keys_to_remove = ['__header__', '__version__', '__globals__']
        for key in keys_to_remove:
            if key in mat_contents:
                del mat_contents[key]

        if var_name:
            if var_name not in mat_contents:
                raise KeyError(f"文件 {file_path} 中未找到变量 '{var_name}'。")
            return mat_contents[var_name]

        # 如果没有指定变量名，且只剩一个键，则直接返回该变量的值
        if len(mat_contents) == 1:
            return next(iter(mat_contents.values()))

        # 否则返回整个字典
        return mat_contents

    except Exception as e:
        raise IOError(f"加载 .mat 文件失败 ({file_path}): {e}")


# 假设也需要一个加载其他数据的通用函数
load_eeg_data = load_mat_matrix
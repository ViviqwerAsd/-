import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

# 默认数据路径
_CIFAR_DIR = Path(__file__).parent / "data"

# 均值与标准差 (按通道统计，形状为 (1, 3, 1, 1) 便于广播)
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)


def _load_batches(file_list: List[str], root: Path = _CIFAR_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    从磁盘加载多个 CIFAR 批次文件

    Args:
        file_list: 批次文件名列表（如 ["data_batch_1", ..., "data_batch_5"]）
        root: 数据根目录

    Returns:
        images: (N, 3072) 的图像数据
        labels: (N,) 的整型标签
    """
    if not root.exists():
        raise FileNotFoundError(f"Cannot locate CIFAR data directory: {root}")

    images, labels = [], []
    for fname in tqdm(file_list, desc="Reading CIFAR batches"):
        fpath = root / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing file: {fpath}")

        with open(fpath, "rb") as f:
            entry = pickle.load(f, encoding="bytes")

        images.append(entry[b"data"].astype(np.float32))
        labels.append(np.array(entry[b"labels"], dtype=np.int64))

    return np.vstack(images), np.concatenate(labels)


def _normalize_cifar(X: np.ndarray) -> np.ndarray:
    """
    图像数据标准化（按通道）

    Args:
        X: 输入图像数据 (N, 3072)

    Returns:
        标准化图像 (N, 3072)
    """
    X = X.reshape(-1, 3, 32, 32)  # 变为 NCHW 格式
    X = (X / 255.0 - _MEAN) / _STD
    return X.reshape(-1, 3072)


def cifar10_dataset(
    train_files: List[str],
    test_file: str = "test_batch",
    root: Path = _CIFAR_DIR
) -> Dict[str, np.ndarray]:
    """
    加载并预处理 CIFAR-10 数据集

    Args:
        train_files: 训练批次文件列表
        test_file: 测试集文件名，默认 "test_batch"
        root: 数据文件存放目录

    Returns:
        包含标准化训练集和测试集的字典
    """
    X_train, y_train = _load_batches(train_files, root)
    X_test, y_test = _load_batches([test_file], root)

    print(f"[INFO] Training samples: {X_train.shape[0]}, Classes: {len(np.unique(y_train))}")
    print(f"[INFO] Test samples: {X_test.shape[0]}, Classes: {len(np.unique(y_test))}")

    return {
        "train_X": _normalize_cifar(X_train),
        "train_y": y_train,
        "test_X": _normalize_cifar(X_test),
        "test_y": y_test
    }

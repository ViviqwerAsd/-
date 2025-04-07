import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)

def load_cifar_batches(
    batch_names: List[str], 
    data_dir: Path = DEFAULT_DATA_DIR
) -> Tuple[np.ndarray, np.ndarray]:
    """加载CIFAR-10批次数据
    
    Args:
        batch_names: 数据批次文件名列表 (e.g. ['data_batch_1', ...])
        data_dir: 数据存储目录路径
        
    Returns:
        X: 图像数据 (N, 3072)
        y: 标签数据 (N,)
        
    Raises:
        FileNotFoundError: 当数据文件不存在时
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    X_list, y_list = [], []
    for batch_name in tqdm(batch_names, desc="Loading batches"):
        batch_path = data_dir / batch_name
        if not batch_path.exists():
            raise FileNotFoundError(f"CIFAR batch not found: {batch_path}")
            
        with batch_path.open('rb') as f:
            batch = pickle.load(f, encoding='bytes')
            
        X_list.append(batch[b'data'].astype(np.float32))
        y_list.append(np.array(batch[b'labels'], dtype=np.int64))

    return np.concatenate(X_list), np.concatenate(y_list)

def preprocess_cifar(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """数据预处理流水线
    
    Args:
        X: 原始图像数据 (N, 3072)
        y: 原始标签数据 (N,)
        
    Returns:
        预处理后的数据:
        - X: 归一化后的图像数据 (N, 3072)
        - y: 标签数据 (N,)
    """
    # 数据标准化 (逐通道)
    X = X.reshape(-1, 3, 32, 32)  # 转换为NCHW格式
    X = (X / 255.0 - CIFAR_MEAN) / CIFAR_STD
    X = X.reshape(-1, 3072)  # 恢复为展平格式

    return X, y

def load_dataset(
    train_batch_names: List[str],
    test_batch_name: str = "test_batch",
    data_dir: Path = DEFAULT_DATA_DIR
) -> Dict[str, np.ndarray]:
    """完整数据加载流程
    
    Args:
        train_batch_names: 训练批次文件名列表 (e.g. ['data_batch_1', ..., 'data_batch_5'])
        test_batch_name: 测试批次文件名，默认为'test_batch'
        data_dir: 数据存储目录路径
        
    Returns:
        包含预处理后的训练集和测试集的字典:
        {
            'train_X': 训练集数据 (N_train, 3072),
            'train_y': 训练集标签 (N_train,),
            'test_X': 测试集数据 (N_test, 3072),
            'test_y': 测试集标签 (N_test,)
        }
    """
    # 加载训练集
    train_X, train_y = load_cifar_batches(train_batch_names, data_dir)
    print(f"成功加载训练集: 总样本数={len(train_X):,}, 类别数={len(np.unique(train_y))}")
    
    # 加载测试集
    test_X, test_y = load_cifar_batches([test_batch_name], data_dir)
    print(f"成功加载测试集: 总样本数={len(test_X):,}, 类别数={len(np.unique(test_y))}")
    
    # 数据预处理
    train_X, train_y = preprocess_cifar(train_X, train_y)
    test_X, test_y = preprocess_cifar(test_X, test_y)
    
    return {
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y
    }
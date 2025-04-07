# -*- coding: utf-8 -*-
import argparse
import numpy as np
from train import train
from test import test
from param_search import param_search
from data_loader import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--test', action='store_true', help="Run testing")
    parser.add_argument('--param_search', action='store_true', help="Run parameter search")
    args = parser.parse_args()

    # 加载数据集
    dataset = load_dataset(
        train_batch_names=[f"data_batch_{i}" for i in range(1, 6)],
        test_batch_name="test_batch"
    )
    X_train, y_train = dataset['train_X'], dataset['train_y']
    X_test, y_test = dataset['test_X'], dataset['test_y']

    if args.train:
        # 训练模型
        train(
            X_train, y_train,
            X_test, y_test, 
            hidden_size=126,
            learning_rate=0.01,
            reg=0.01,
            dropout_rate=0
        )
    elif args.test:
        # 测试模型
        test('/data2/WangXinyi/homework/hw1/cv_best_model.npz', X_test, y_test)
    elif args.param_search:
        # 参数搜索
        param_search(X_train, y_train, X_test, y_test)
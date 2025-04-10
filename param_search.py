import itertools
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from train import train

def grid_search(X_train, y_train, X_val, y_val):
    """
    执行网格搜索以选择最佳超参数组合
    """
    # 定义待调节的超参数范围
    search_space = {
        'hidden_size': [128, 256, 512, 1024],
        'learning_rate': [0.1, 0.01, 0.001],
        'reg': [0.01, 0.1],
        'dropout_rate': [0.2, 0.5],
        'activation': ['relu', 'sigmoid']
    }

    all_results = []
    max_accuracy = 0.0
    optimal_setting = {}
    
    param_names = list(search_space.keys())
    param_values = list(search_space.values())

    for config in itertools.product(*param_values):
        current_config = dict(zip(param_names, config))
        print(f"Running config: {current_config}")
        
        model_state, acc = train(X_train, y_train, X_val, y_val, **current_config)
        
        # 保存当前配置与结果
        record = current_config.copy()
        record['val_acc'] = acc
        all_results.append(record)
        
        if acc > max_accuracy:
            max_accuracy = acc
            optimal_setting = current_config.copy()
            np.savez('cv_best_model.npz',
                     W1=model_state['W1'],
                     b1=model_state['b1'],
                     W2=model_state['W2'],
                     b2=model_state['b2'],
                     hidden_size=current_config['hidden_size'],
                     activation=current_config['activation'],
                     dropout_rate=current_config['dropout_rate'])
    
    # 写入所有超参数组合的验证结果
    with open('param_search_results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=param_names + ['val_acc'])
        writer.writeheader()
        writer.writerows(all_results)
    
    print("\n>> Best Config Found:")
    print(optimal_setting)
    print(f"Validation Accuracy: {max_accuracy:.4f}")
    
    plot_search_results(all_results)

def plot_search_results(results):
    """
    可视化超参数搜索的部分趋势
    """
    df = pd.DataFrame(results)

    # 图1：不同学习率下，隐藏层大小对准确率的影响
    plt.figure(figsize=(12, 5))
    for lr_val in df['learning_rate'].unique():
        filtered = df[df['learning_rate'] == lr_val]
        plt.plot(filtered['hidden_size'], filtered['val_acc'], marker='o', label=f"LR={lr_val}")
    plt.title("Hidden Size vs Accuracy (Different LR)")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("search_hidden_vs_lr.png")
    plt.show()

    # 图2：不同正则强度下，隐藏层大小对准确率的影响
    plt.figure(figsize=(12, 5))
    for reg_val in df['reg'].unique():
        filtered = df[df['reg'] == reg_val]
        plt.plot(filtered['hidden_size'], filtered['val_acc'], marker='o', label=f"Reg={reg_val}")
    plt.title("Hidden Size vs Accuracy (Different Reg)")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("search_hidden_vs_reg.png")
    plt.show()

    # 图3：激活函数对准确率的影响
    plt.figure(figsize=(12, 5))
    for act in df['activation'].unique():
        filtered = df[df['activation'] == act]
        plt.plot(filtered['hidden_size'], filtered['val_acc'], marker='o', label=f"Act={act}")
    plt.title("Hidden Size vs Accuracy (Different Activation)")
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("search_hidden_vs_act.png")
    plt.show()

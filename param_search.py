# import itertools
# from train import train
# import numpy as np

# def param_search(X_train, y_train, X_val, y_val):
#     params = {
#         'hidden_size': [128,256,512,1024], # 确保参数名与训练函数一致
#         'learning_rate': [0.1,0.01,0.001],
#         'reg': [0.01, 0.1],
#         'dropout_rate': [0.2, 0.5]
#     }
    
#     best_acc = 0
#     best_params = {}
    
#     # 生成所有参数组合的键和值
#     param_names = params.keys()
#     param_values = params.values()
    
#     # 遍历所有参数组合
#     for combination in itertools.product(*param_values):
#         current_params = dict(zip(param_names, combination))
#         print(f"Training with {current_params}")
        
#         # 解包参数并训练模型
#         model_params, val_acc = train(X_train, y_train, X_val, y_val, **current_params)
        
#         if val_acc > best_acc:
#             best_acc = val_acc
#             best_params = current_params.copy()  # 保存深拷贝避免引用问题
#             best_model_params = model_params
#             np.savez('cv_best_model.npz', 
#                     W1=best_model_params['W1'],
#                     b1=best_model_params['b1'],
#                     W2=best_model_params['W2'],
#                     b2=best_model_params['b2'],
#                     hidden_size=best_params['hidden_size'])
    
#     print("Best parameters:", best_params)
#     print("Best validation accuracy:", best_acc)

import itertools
import csv
import matplotlib.pyplot as plt
import pandas as pd
from train import train
import numpy as np

def param_search(X_train, y_train, X_val, y_val):
    params = {
        'hidden_size': [128, 256, 512, 1024],
        'learning_rate': [0.1, 0.01, 0.001],
        'reg': [0.01, 0.1],
        'dropout_rate': [0.2, 0.5],
        'activation': ['relu', 'sigmoid'] 
    }
    
    best_acc = 0
    best_params = {}
    results = []  # 用于记录所有实验结果
    
    # 生成所有参数组合的键和值
    param_names = params.keys()
    param_values = params.values()
    
    # 遍历所有参数组合
    for combination in itertools.product(*param_values):
        current_params = dict(zip(param_names, combination))
        print(f"Training with {current_params}")
        
        # 解包参数并训练模型
        model_params, val_acc = train(X_train, y_train, X_val, y_val, **current_params)
        
        # 记录当前实验结果
        results.append({**current_params, 'val_acc': val_acc})
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = current_params.copy()  # 保存深拷贝避免引用问题
            best_model_params = model_params
            np.savez('cv_best_model.npz', 
                    W1=best_model_params['W1'],
                    b1=best_model_params['b1'],
                    W2=best_model_params['W2'],
                    b2=best_model_params['b2'],
                    hidden_size=best_params['hidden_size'])
    
    # 将结果保存到CSV文件
    with open('param_search_results.csv', mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=params.keys() | {'val_acc'})
        writer.writeheader()
        writer.writerows(results)
    
    print("Best parameters:", best_params)
    print("Best validation accuracy:", best_acc)
    
    # 可视化超参数的影响
    visualize_param_search_results(results)

def visualize_param_search_results(results):
    """可视化超参数搜索结果"""
    results_df = pd.DataFrame(results)
    
    # 绘制学习率 vs 验证集准确率
    plt.figure(figsize=(12, 6))
    for lr in results_df['learning_rate'].unique():
        subset = results_df[results_df['learning_rate'] == lr]
        plt.plot(subset['hidden_size'], subset['val_acc'], label=f'lr={lr}')
    plt.xlabel('Hidden Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Hidden Size for Different Learning Rates')
    plt.legend()
    plt.savefig('param_search_learning_rate_vs_hidden_size.png')
    plt.show()
    
    # 绘制正则化强度 vs 验证集准确率
    plt.figure(figsize=(12, 6))
    for reg in results_df['reg'].unique():
        subset = results_df[results_df['reg'] == reg]
        plt.plot(subset['hidden_size'], subset['val_acc'], label=f'reg={reg}')
    plt.xlabel('Hidden Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Hidden Size for Different Regularization Strengths')
    plt.legend()
    plt.savefig('param_search_reg_vs_hidden_size.png')
    plt.show()
    
    # 绘制激活函数 vs 验证集准确率
    plt.figure(figsize=(12, 6))
    for activation in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == activation]
        plt.plot(subset['hidden_size'], subset['val_acc'], label=f'activation={activation}')
    plt.xlabel('Hidden Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Hidden Size for Different Activation Functions')
    plt.legend()
    plt.savefig('param_search_activation_vs_hidden_size.png')
    plt.show()
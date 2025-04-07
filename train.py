import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNet

def evaluate(model, X, y):
    """评估模型在给定数据上的准确率"""
    model.eval()  # 设置为评估模式(不应用dropout)
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    model.train()  # 恢复训练模式
    return np.mean(predictions == y)

def compute_loss(model, X, y, reg):
    """计算模型的损失值"""
    model.eval()  # 设置为评估模式(不应用dropout)
    probs = model.forward(X)
    data_loss = -np.log(probs[range(len(y)), y]).mean()
    reg_loss = 0.5 * reg * (np.sum(model.params['W1']**2) + np.sum(model.params['W2']**2))
    model.train()  # 恢复训练模式
    return data_loss + reg_loss

def train(
    X_train, y_train, 
    X_test, y_test,  # 替换 X_val 和 y_val
    hidden_size=1024, activation='relu',
    reg=0.01, learning_rate=1e-3,
    epochs=1000, batch_size=200,
    lr_decay=0.9, lr_decay_every=5, 
    early_stop_step=20, dropout_rate=0.2
):
    """训练模型并使用测试集评估性能"""
    input_dim = X_train.shape[1]
    model = ThreeLayerNet(input_dim, hidden_size, 10, activation, dropout_rate)
    model.train()  # 设置为训练模式
    best_test_acc = 0.0
    no_improvement_count = 0
    
    # 初始化记录变量
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    recorded_epochs = []  # 记录实际epoch数的列表
    
    for epoch in range(epochs):
        # 学习率衰减
        if epoch % lr_decay_every == 0 and epoch > 0:
            learning_rate *= lr_decay
        
        # 随机打乱数据
        shuffle_idx = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[shuffle_idx]
        y_shuffled = y_train[shuffle_idx]
        
        # 小批量训练
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # 前向传播(自动应用dropout)
            probs = model.forward(X_batch)
            
            # 计算损失
            data_loss = -np.log(probs[range(len(y_batch)), y_batch]).mean()
            reg_loss = 0.5 * reg * (np.sum(model.params['W1']**2) + np.sum(model.params['W2']**2))
            
            # 反向传播
            grads = model.backward(X_batch, y_batch, reg)
            
            # 参数更新（SGD）
            for param in model.params:
                model.params[param] -= learning_rate * grads[param]
        
        # 使用测试集评估模型性能
        test_acc = evaluate(model, X_test, y_test)
        print(f"Epoch {epoch+1}/{epochs} | Test Acc: {test_acc:.4f}")

        # 每10个epoch记录一次数据
        if (epoch+1) % 10 == 0 or epoch == 0:
            train_loss = compute_loss(model, X_train, y_train, reg)
            test_loss = compute_loss(model, X_test, y_test, reg)
            train_acc = evaluate(model, X_train, y_train)
            
            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            recorded_epochs.append(epoch + 1)
            
            print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        # 保存最佳模型及其训练曲线
        if test_acc > best_test_acc:
            no_improvement_count = 0
            best_test_acc = test_acc
            best_params = {
                'W1': model.params['W1'].copy(),
                'b1': model.params['b1'].copy(),
                'W2': model.params['W2'].copy(),
                'b2': model.params['b2'].copy()
            }
            np.savez('best_model.npz', 
                    W1=best_params['W1'],
                    b1=best_params['b1'],
                    W2=best_params['W2'],
                    b2=best_params['b2'],
                    hidden_size=hidden_size,
                    activation=activation,
                    dropout_rate=dropout_rate)
            
            # 保存最佳模型的训练曲线
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(recorded_epochs, train_loss_history, 'o-', label='Train Loss')
            plt.plot(recorded_epochs, test_loss_history,'o-',  label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(recorded_epochs, train_acc_history,'o-',  label='Train Accuracy')
            plt.plot(recorded_epochs, test_acc_history,'o-',  label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Test Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('best_model_training_curves.png')
            plt.close()
        else:
            no_improvement_count += 1
            if no_improvement_count > early_stop_step:
                print(f"\nEarly stopping triggered after {epoch+1} epochs without improvement.")
                break
    
    # 绘制loss曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(recorded_epochs, train_loss_history, 'o-', label='Train Loss')
    plt.plot(recorded_epochs, test_loss_history,'o-',  label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recorded_epochs, train_acc_history,'o-',  label='Train Accuracy')
    plt.plot(recorded_epochs, test_acc_history,'o-',  label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    return best_params, best_test_acc
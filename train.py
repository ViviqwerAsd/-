import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNet

def compute_loss(model, X, y, reg):
    """返回当前模型在给定数据和正则项下的总损失"""
    model.eval()  # 切换为评估模式，避免dropout干扰
    probs = model.forward(X)
    core_loss = -np.log(probs[np.arange(len(y)), y]).mean()
    weight_decay = 0.5 * reg * (np.sum(model.params['W1']**2) + np.sum(model.params['W2']**2))
    model.train()  # 恢复训练模式
    return core_loss + weight_decay

def evaluate(model, X, y):
    """计算预测准确率"""
    model.eval()
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    model.train()
    return np.mean(preds == y)

def train(
    X_train, y_train, 
    X_test, y_test,
    hidden_size=1024, activation='relu',
    reg=0.01, learning_rate=1e-3,
    epochs=1000, batch_size=200,
    lr_decay=0.9, lr_decay_every=5,
    early_stop_step=20, dropout_rate=0.2
):
    """训练神经网络并使用测试集评估性能"""
    
    input_dim = X_train.shape[1]
    model = ThreeLayerNet(input_dim, hidden_size, 10, activation, dropout_rate)
    model.train()
    
    best_test_acc = 0.0
    no_improvement = 0
    
    # 训练过程记录
    recorded_epochs = []
    train_loss_log, test_loss_log = [], []
    train_acc_log, test_acc_log = [], []

    for epoch in range(epochs):
        # 每隔指定轮数衰减学习率
        if epoch > 0 and epoch % lr_decay_every == 0:
            learning_rate *= lr_decay
        
        # 打乱训练数据顺序
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        # 小批量训练
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            probs = model.forward(X_batch)
            core_loss = -np.log(probs[np.arange(len(y_batch)), y_batch]).mean()
            reg_loss = 0.5 * reg * (np.sum(model.params['W1']**2) + np.sum(model.params['W2']**2))

            grads = model.backward(X_batch, y_batch, reg)

            for param in model.params:
                model.params[param] -= learning_rate * grads[param]

        # 当前测试集准确率
        test_acc = evaluate(model, X_test, y_test)
        print(f"Epoch {epoch+1}/{epochs} | Test Acc: {test_acc:.4f}")

        # 每10轮或第一轮记录一次训练过程
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_loss = compute_loss(model, X_train, y_train, reg)
            test_loss = compute_loss(model, X_test, y_test, reg)
            train_acc = evaluate(model, X_train, y_train)

            recorded_epochs.append(epoch + 1)
            train_loss_log.append(train_loss)
            test_loss_log.append(test_loss)
            train_acc_log.append(train_acc)
            test_acc_log.append(test_acc)

            print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        # 若当前模型表现为最优，则保存
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improvement = 0

            best_params = {
                'W1': model.params['W1'].copy(),
                'b1': model.params['b1'].copy(),
                'W2': model.params['W2'].copy(),
                'b2': model.params['b2'].copy()
            }

            # 保存最优模型
            np.savez('best_model.npz', 
                     W1=best_params['W1'],
                     b1=best_params['b1'],
                     W2=best_params['W2'],
                     b2=best_params['b2'],
                     hidden_size=hidden_size,
                     activation=activation,
                     dropout_rate=dropout_rate)

            # 可视化保存
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(recorded_epochs, train_loss_log, 'o-', label='Train Loss')
            plt.plot(recorded_epochs, test_loss_log, 'o-', label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(recorded_epochs, train_acc_log, 'o-', label='Train Acc')
            plt.plot(recorded_epochs, test_acc_log, 'o-', label='Test Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Curve')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('best_model_training_curves.png')
            plt.close()
        else:
            no_improvement += 1
            if no_improvement > early_stop_step:
                print(f"\nEarly stopping at epoch {epoch+1} due to no improvement.")
                break

    # 最终完整曲线绘图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(recorded_epochs, train_loss_log, 'o-', label='Train Loss')
    plt.plot(recorded_epochs, test_loss_log, 'o-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Final Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recorded_epochs, train_acc_log, 'o-', label='Train Acc')
    plt.plot(recorded_epochs, test_acc_log, 'o-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Final Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print(f"\nBest Test Accuracy Achieved: {best_test_acc:.4f}")
    return best_params, best_test_acc

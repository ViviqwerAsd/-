import numpy as np

class NeuralNetThreeLayer:
    def __init__(self, input_size, hidden_dim, output_size, act_fn='relu', dropout_p=0.5):
        # 模型参数初始化
        self.params = {
            'W1': np.random.randn(input_size, hidden_dim) * np.sqrt(2.0 / input_size),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, output_size) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros(output_size)
        }
        self.activation = act_fn
        self.dropout_p = dropout_p
        self.training = True  # 默认处于训练模式

    def set_training(self, mode=True):
        """切换训练/评估模式"""
        self.training = mode

    def predict(self, X):
        """
        前向传播阶段，返回类别概率
        """
        # 第一层线性+激活
        z1 = np.dot(X, self.params['W1']) + self.params['b1']
        if self.activation == 'relu':
            a1 = np.maximum(0, z1)
        elif self.activation == 'sigmoid':
            a1 = 1.0 / (1.0 + np.exp(-z1))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        # Dropout只在训练时使用
        if self.training:
            self.dropout_mask = (np.random.rand(*a1.shape) > self.dropout_p) / (1.0 - self.dropout_p)
            a1 *= self.dropout_mask
        self.a1 = a1  # 存储用于反向传播
        self.z1 = z1

        # 第二层线性+softmax
        z2 = np.dot(a1, self.params['W2']) + self.params['b2']
        z2_shifted = z2 - np.max(z2, axis=1, keepdims=True)  # 稳定性处理
        exp_scores = np.exp(z2_shifted)
        self.probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.z2 = z2  # 存储用于梯度计算
        return self.probabilities

    def compute_grads(self, X, y, l2_reg):
        """
        反向传播，返回梯度字典
        """
        grads = {}
        num_samples = X.shape[0]

        # 输出层误差
        delta_output = self.probabilities.copy()
        delta_output[np.arange(num_samples), y] -= 1
        delta_output /= num_samples

        # W2和b2的梯度
        grads['W2'] = np.dot(self.a1.T, delta_output) + l2_reg * self.params['W2']
        grads['b2'] = np.sum(delta_output, axis=0)

        # 反向传播到隐藏层
        delta_hidden = np.dot(delta_output, self.params['W2'].T)

        if self.training:
            delta_hidden *= self.dropout_mask  # dropout反向处理

        if self.activation == 'relu':
            delta_hidden[self.z1 <= 0] = 0
        elif self.activation == 'sigmoid':
            delta_hidden *= self.a1 * (1.0 - self.a1)

        grads['W1'] = np.dot(X.T, delta_hidden) + l2_reg * self.params['W1']
        grads['b1'] = np.sum(delta_hidden, axis=0)
        return grads

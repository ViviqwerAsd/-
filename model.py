import numpy as np

class ThreeLayerNet:
    def __init__(self, input_dim, hidden_size, output_dim, activation='relu', dropout_rate=0.5):
        self.params = {}
        self.params['W1'] = np.random.randn(input_dim, hidden_size) * np.sqrt(2. / input_dim)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_dim) * np.sqrt(2. / hidden_size)
        self.params['b2'] = np.zeros(output_dim)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.mode = 'train'  # 默认训练模式

    def forward(self, X):
        # 第一层前向传播
        self.z1 = np.dot(X, self.params['W1']) + self.params['b1']
        if self.activation == 'relu':
            self.a1 = np.maximum(0, self.z1)
        elif self.activation == 'sigmoid':
            self.a1 = 1 / (1 + np.exp(-self.z1))
        
        # 只在训练时应用dropout
        if self.mode == 'train':
            # 生成dropout mask
            self.dropout_mask = (np.random.rand(*self.a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a1 *= self.dropout_mask
        
        # 第二层前向传播
        self.z2 = np.dot(self.a1, self.params['W2']) + self.params['b2']
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, reg):
        grads = {}
        num_samples = X.shape[0]
        delta3 = self.probs.copy()
        delta3[range(num_samples), y] -= 1
        delta3 /= num_samples
        
        grads['W2'] = np.dot(self.a1.T, delta3) + reg * self.params['W2']
        grads['b2'] = np.sum(delta3, axis=0)
        
        delta2 = np.dot(delta3, self.params['W2'].T)
        # 如果在训练时应用了dropout，反向传播时也要考虑
        if self.mode == 'train':
            delta2 *= self.dropout_mask
        
        if self.activation == 'relu':
            delta2[self.z1 <= 0] = 0
        elif self.activation == 'sigmoid':
            delta2 *= self.a1 * (1 - self.a1)
        
        grads['W1'] = np.dot(X.T, delta2) + reg * self.params['W1']
        grads['b1'] = np.sum(delta2, axis=0)
        return grads
    
    def train(self):
        """设置模型为训练模式"""
        self.mode = 'train'
    
    def eval(self):
        """设置模型为评估模式"""
        self.mode = 'eval'
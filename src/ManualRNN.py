import numpy as np

class ManualRNN:
    def __init__(self, input_dim, hidden_dim=8, output_dim=1):
        """
        适配每个时间步有标签的手动RNN
        :param input_dim: 特征维度（如3：GPA, DCCY, JXJ）
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度（1：二分类概率）
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始化权重（与之前一致）
        self.W_x = np.random.randn(hidden_dim, input_dim) * 0.01  # 输入→隐藏层
        self.W_h = np.random.randn(hidden_dim, hidden_dim) * 0.01  # 隐藏层→隐藏层（记忆）
        self.W_y = np.random.randn(output_dim, hidden_dim) * 0.01  # 隐藏层→输出层
        # 偏置
        self.b_h = np.zeros((hidden_dim, 1))  # 隐藏层偏置
        self.b_y = np.zeros((output_dim, 1))  # 输出层偏置
        
        # 记录梯度（累积每个时间步的梯度）
        self.grad_W_x = np.zeros_like(self.W_x)
        self.grad_W_h = np.zeros_like(self.W_h)
        self.grad_W_y = np.zeros_like(self.W_y)
        self.grad_b_h = np.zeros_like(self.b_h)
        self.grad_b_y = np.zeros_like(self.b_y)


    def forward(self, x_seq):
        """
        前向传播：返回每个时间步的预测概率和隐藏状态
        :param x_seq: 单个时序序列，shape=(seq_len, input_dim)
        :return: y_preds: 每个时间步的预测概率，shape=(seq_len,)
                 h_states: 每个时间步的隐藏状态，list of (hidden_dim, 1)
        """
        seq_len = x_seq.shape[0]
        h_prev = np.zeros((self.hidden_dim, 1))  # 初始隐藏状态（全0）
        h_states = []  # 存储每个时间步的隐藏状态
        y_preds = []   # 存储每个时间步的预测概率
        
        for t in range(seq_len):
            x_t = x_seq[t].reshape(-1, 1)  # 当前时间步输入，(input_dim, 1)
            # 计算当前隐藏状态（记忆上一时间步信息）
            h_t = np.tanh(self.W_x @ x_t + self.W_h @ h_prev + self.b_h)
            h_states.append(h_t)
            # 计算当前时间步的输出（预测概率）
            y_t = 1 / (1 + np.exp(-(self.W_y @ h_t + self.b_y)))  # sigmoid激活
            y_preds.append(y_t[0, 0])  # 存储标量概率
            # 更新隐藏状态（传递到下一时间步）
            h_prev = h_t
        
        return np.array(y_preds), h_states  # 返回所有时间步的预测和隐藏状态


    def backward(self, x_seq, h_states, y_preds, y_trues):
        """
        反向传播：累积每个时间步的梯度（BPTT）
        :param x_seq: 输入序列，(seq_len, input_dim)
        :param h_states: 前向传播的隐藏状态，list of (hidden_dim, 1)
        :param y_preds: 每个时间步的预测概率，(seq_len,)
        :param y_trues: 每个时间步的真实标签，(seq_len,)
        """
        seq_len = x_seq.shape[0]
        # 重置梯度（每次反向传播前清零）
        self.grad_W_x.fill(0)
        self.grad_W_h.fill(0)
        self.grad_W_y.fill(0)
        self.grad_b_h.fill(0)
        self.grad_b_y.fill(0)
        
        # 初始化最后一个时间步的梯度传递（从输出层开始）
        dh_next = np.zeros((self.hidden_dim, 1))  # 初始为0
        
        # 从最后一个时间步反向遍历到第一个
        for t in reversed(range(seq_len)):
            # 当前时间步的预测、真实标签和隐藏状态
            y_pred = y_preds[t]
            y_true = y_trues[t]
            h_t = h_states[t]  # (hidden_dim, 1)
            x_t = x_seq[t].reshape(-1, 1)  # (input_dim, 1)
            
            # 1. 计算当前时间步输出层的梯度
            dy = (y_pred - y_true).reshape(1, 1)  # 损失对输出的导数，(1,1)
            # 累积输出层权重和偏置的梯度
            self.grad_W_y += dy @ h_t.T  # (1,1) @ (1, hidden_dim) → (1, hidden_dim)
            self.grad_b_y += dy  # (1,1)
            
            # 2. 计算当前时间步隐藏层的梯度（结合输出层和下一时间步的梯度）
            # 隐藏层梯度 = 输出层传递的梯度 + 下一时间步隐藏层传递的梯度
            dh = self.W_y.T @ dy + dh_next  # (hidden_dim,1) + (hidden_dim,1) → (hidden_dim,1)
            # tanh的导数：1 - tanh^2(h_t)
            dtanh = (1 - h_t**2) * dh  # (hidden_dim,1)
            
            # 3. 累积隐藏层相关的梯度
            self.grad_b_h += dtanh  # 偏置梯度
            self.grad_W_x += dtanh @ x_t.T  # 输入→隐藏层权重梯度 (hidden_dim, input_dim)
            # 隐藏层→隐藏层权重梯度（依赖上一时间步的隐藏状态）
            h_prev = h_states[t-1] if t > 0 else np.zeros_like(h_t)  # (hidden_dim,1)
            self.grad_W_h += dtanh @ h_prev.T  # (hidden_dim, hidden_dim)
            
            # 4. 将梯度传递到上一个时间步
            dh_next = self.W_h.T @ dtanh  # (hidden_dim, hidden_dim) @ (hidden_dim,1) → (hidden_dim,1)


    def update_weights(self, lr=0.01, clip_value=0.5):
        """用梯度下降更新权重（与之前一致，带梯度裁剪）"""
        # 梯度裁剪（防止爆炸）
        for grad in [self.grad_W_x, self.grad_W_h, self.grad_W_y, self.grad_b_h, self.grad_b_y]:
            np.clip(grad, -clip_value, clip_value, out=grad)
        
        # 梯度下降更新
        self.W_x -= lr * self.grad_W_x
        self.W_h -= lr * self.grad_W_h
        self.W_y -= lr * self.grad_W_y
        self.b_h -= lr * self.grad_b_h
        self.b_y -= lr * self.grad_b_y
import numpy as np

class ManualRNN:
    def __init__(self, input_dim, hidden_dim=8, output_dim=1):
        """
        手动RNN：捕捉时序依赖
        :param input_dim: 特征维度（如3：GPA, DCCY, JXJ）
        :param hidden_dim: 隐藏层维度（短序列用8-16即可）
        :param output_dim: 输出维度（1：二分类概率）
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始化权重（随机小值，避免激活函数饱和）
        self.W_x = np.random.randn(hidden_dim, input_dim) * 0.01  # 输入→隐藏层权重
        self.W_h = np.random.randn(hidden_dim, hidden_dim) * 0.01  # 隐藏层→隐藏层权重（记忆核心）
        self.W_y = np.random.randn(output_dim, hidden_dim) * 0.01  # 隐藏层→输出层权重
        # 偏置
        self.b_h = np.zeros((hidden_dim, 1))  # 隐藏层偏置
        self.b_y = np.zeros((output_dim, 1))  # 输出层偏置
        
        # 记录梯度（用于反向传播更新）
        self.grad_W_x = np.zeros_like(self.W_x)
        self.grad_W_h = np.zeros_like(self.W_h)
        self.grad_W_y = np.zeros_like(self.W_y)
        self.grad_b_h = np.zeros_like(self.b_h)
        self.grad_b_y = np.zeros_like(self.b_y)


    def forward(self, x_seq):
        """
        前向传播：计算序列的输出和隐藏状态
        :param x_seq: 单个时序序列，shape=(seq_len, input_dim)
        :return: y_pred: 最后一个时间步的输出概率（标量）
                 h_states: 所有时间步的隐藏状态列表，len=seq_len
        """
        seq_len = x_seq.shape[0]
        h_prev = np.zeros((self.hidden_dim, 1))  # 初始隐藏状态（全0）
        h_states = []  # 存储每个时间步的隐藏状态
        
        for t in range(seq_len):
            x_t = x_seq[t].reshape(-1, 1)  # 当前时间步输入，shape=(input_dim, 1)
            # 隐藏状态计算：h_t = tanh(W_x·x_t + W_h·h_prev + b_h)
            h_t = np.tanh(self.W_x @ x_t + self.W_h @ h_prev + self.b_h)
            h_states.append(h_t)
            h_prev = h_t  # 更新隐藏状态（传递到下一时间步）
        
        # 输出层：用最后一个时间步的隐藏状态计算预测概率（sigmoid激活）
        y_pred = 1 / (1 + np.exp(-(self.W_y @ h_prev + self.b_y)))  # sigmoid：输出0-1概率
        return y_pred[0, 0], h_states  # y_pred为标量，h_states为列表


    def backward(self, x_seq, h_states, y_pred, y_true):
        """
        反向传播（BPTT：沿时间反向传播）：计算权重梯度
        :param x_seq: 输入序列，shape=(seq_len, input_dim)
        :param h_states: 前向传播的隐藏状态列表
        :param y_pred: 预测概率（标量）
        :param y_true: 真实标签（0或1）
        """
        seq_len = x_seq.shape[0]
        # 重置梯度
        self.grad_W_x.fill(0)
        self.grad_W_h.fill(0)
        self.grad_W_y.fill(0)
        self.grad_b_h.fill(0)
        self.grad_b_y.fill(0)
        
        # 1. 输出层梯度（二分类交叉熵损失的导数）
        # 损失函数：L = -y_true*log(y_pred) - (1-y_true)*log(1-y_pred)
        dy = (y_pred - y_true)  # 损失对输出y的导数（简化计算）
        self.grad_W_y += dy * h_states[-1].T  # 最后一个时间步的隐藏状态
        self.grad_b_y += dy
        
        # 2. 沿时间反向传播（从最后一个时间步到第一个）
        dh_next = self.W_y.T @ dy  # 输出层梯度传递到隐藏层（最后一步）
        for t in reversed(range(seq_len)):
            h_t = h_states[t]
            x_t = x_seq[t].reshape(-1, 1)  # 当前时间步输入
            # tanh的导数：1 - tanh^2(h_t)
            dtanh = (1 - h_t**2) * dh_next
            # 累积隐藏层偏置梯度
            self.grad_b_h += dtanh
            # 累积输入→隐藏层权重梯度
            self.grad_W_x += dtanh @ x_t.T
            # 累积隐藏层→隐藏层权重梯度（依赖上一时间步的隐藏状态）
            h_prev = h_states[t-1] if t > 0 else np.zeros_like(h_t)
            self.grad_W_h += dtanh @ h_prev.T
            # 传递梯度到上一个时间步
            dh_next = self.W_h.T @ dtanh


    def update_weights(self, lr=0.01, clip_value=0.5):
        """
        用梯度下降更新权重（带梯度裁剪，防止梯度爆炸）
        :param lr: 学习率
        :param clip_value: 梯度裁剪阈值
        """
        # 梯度裁剪（避免梯度爆炸，对短序列尤其重要）
        for grad in [self.grad_W_x, self.grad_W_h, self.grad_W_y, self.grad_b_h, self.grad_b_y]:
            np.clip(grad, -clip_value, clip_value, out=grad)
        
        # 梯度下降更新
        self.W_x -= lr * self.grad_W_x
        self.W_h -= lr * self.grad_W_h
        self.W_y -= lr * self.grad_W_y
        self.b_h -= lr * self.grad_b_h
        self.b_y -= lr * self.grad_b_y

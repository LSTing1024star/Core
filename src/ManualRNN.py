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


    def backward(self, x_seq, h_states, y_preds, y_trues, mask_seq):
        """
        反向传播（仅累积真实时间步的梯度）
        :param mask_seq: 掩码序列，shape=(seq_len,)，1=真实时间步
        """
        seq_len = x_seq.shape[0]
        self.grad_W_x.fill(0)
        self.grad_W_h.fill(0)
        self.grad_W_y.fill(0)
        self.grad_b_h.fill(0)
        self.grad_b_y.fill(0)
        
        dh_next = np.zeros((self.hidden_dim, 1))
        
        for t in reversed(range(seq_len)):
            # 仅处理真实时间步（mask=1）
            if mask_seq[t] == 0:
                # 填充时间步：跳过梯度计算，直接传递空梯度
                h_t = h_states[t]
                dtanh = np.zeros_like(h_t)  # 填充时间步梯度为0
            else:
                # 真实时间步：正常计算梯度
                y_pred = y_preds[t]
                y_true = y_trues[t]
                h_t = h_states[t]
                x_t = x_seq[t].reshape(-1, 1)
                
                # 输出层梯度
                dy = (y_pred - y_true).reshape(1, 1)
                self.grad_W_y += dy @ h_t.T
                self.grad_b_y += dy
                
                # 隐藏层梯度
                dh = self.W_y.T @ dy + dh_next
                dtanh = (1 - h_t**2) * dh
                
                # 累积梯度
                self.grad_b_h += dtanh
                self.grad_W_x += dtanh @ x_t.T
                h_prev = h_states[t-1] if t > 0 else np.zeros_like(h_t)
                self.grad_W_h += dtanh @ h_prev.T
            
            # 传递梯度到上一时间步（无论是否填充，均传递，保证时序连贯性）
            dh_next = self.W_h.T @ dtanh
            

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
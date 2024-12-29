import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
import numpy as np


'''
DQN中的思想：
    经验重演，降低序列的相关性
    目标网络的思想，将评估和选择动作解耦
'''


# 定义 Q 网络模型（用于 DQN 和 Double DQN）
class Model(nn.Module):
    """基本的 Q 网络模型，输入为状态向量，输出为动作的 Q 值"""
    def __init__(self, num_inputs, num_actions):
        """
        初始化 Q 网络

        参数：
            - num_inputs: 状态空间的维度
            - num_actions: 动作空间的数量
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)  # 第一层全连接层
        self.fc2 = nn.Linear(128, 128)        # 第二层全连接层
        self.out = nn.Linear(128, num_actions)  # 输出层（每个动作的 Q 值）

    def forward(self, x):
        """
        前向传播过程

        参数：
            - x: 输入状态
        返回：
            - Q 值向量，每个动作对应一个值
        """
        x = F.relu(self.fc1(x))  # 激活函数 ReLU
        x = F.relu(self.fc2(x))
        return self.out(x)  # 输出 Q 值向量


# 定义用于存储单条经验的类
class Data:
    """存储单条经验的类，包含状态、动作、奖励、下一状态和完成标志"""
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


# 定义经验回放池
class Memory:
    """经验回放池，用于存储与随机采样训练数据"""
    def __init__(self, capacity):
        """
        初始化经验池

        参数：
            - capacity: 经验池的最大容量
        """
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        """
        向经验池中存储一条新经验

        参数：
            - data: 单条经验（Data 类的对象）
        """
        self.buffer.append(data)

    def get(self, batch_size):
        """
        从经验池中随机采样一批经验

        参数：
            - batch_size: 批量大小

        返回：
            - 采样的经验列表
        """
        return random.sample(self.buffer, batch_size)


# 定义 DQN 算法
class DQN:
    """基本的 DQN 算法实现"""
    def __init__(self, num_states, num_actions, device, lr, gamma, capacity):
        """
        初始化 DQN 算法。

        参数：
        - num_states: 状态空间维度
        - num_actions: 动作空间数量
        - device: 计算设备（CPU 或 GPU）
        - lr: 学习率
        - gamma: 折扣因子
        - capacity: 经验池容量
        """
        print('model using device:', device)
        self.eval_net = Model(num_states, num_actions).to(device)  # 评估网络
        self.target_net = Model(num_states, num_actions).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)  # Adam 优化器
        self.memory = Memory(capacity=capacity)  # 初始化经验池
        self.gamma = gamma
        self.device = device
        self.learn_step_counter = 0  # 学习步数计数器

    def choose_action(self, state, epsilon):
        """
        根据 epsilon-greedy 策略选择动作

        参数：
        - state: 当前状态
        - epsilon: 探索概率

        返回：
        - 选择的动作
        """
        if random.random() > epsilon:  # 按贪婪策略选择动作
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.eval_net(state)
            action = torch.argmax(action_values).item()
        else:  # 按随机策略选择动作
            action = random.randint(0, self.eval_net.out.out_features - 1)
        return action

    def learn(self, batch_size):
        """
        更新 Q 网络

        参数：
        - batch_size: 批量大小
        """
        if len(self.memory.buffer) < batch_size:
            return

        # 从经验池中随机采样
        batch = self.memory.get(batch_size)
        states = torch.tensor([d.state for d in batch], dtype=torch.float).to(self.device)
        actions = torch.tensor([d.action for d in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([d.reward for d in batch], dtype=torch.float).to(self.device)
        next_states = torch.tensor([d.next_state for d in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([d.done for d in batch], dtype=torch.float).to(self.device)

        # Q(s, a)
        q_values = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # max Q(s', a')
        next_q_values = self.target_net(next_states).max(1)[0]
        # TD 目标
        target = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = F.mse_loss(q_values, target)

        # 反向传播更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔固定步数同步目标网络
        if self.learn_step_counter % 10 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1


# 定义 Dueling DQN 网络
class DuelingModel(nn.Module):
    """
    Dueling DQN 模型，将 Q 值分解为状态价值 V(s) 和动作优势 A(s, a)
    """
    def __init__(self, num_inputs, num_actions):
        """
        初始化 Dueling DQN

        参数：
        - num_inputs: 状态空间维度
        - num_actions: 动作空间数量
        """
        super(DuelingModel, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)

        # 定义 V 和 A 的分支
        self.value_stream = nn.Linear(128, 1)  # 状态价值 V(s)
        self.advantage_stream = nn.Linear(128, num_actions)  # 动作优势 A(s, a)

    def forward(self, x):
        """
        前向传播过程

        参数：
            - x: 输入状态
        返回：
            - Q 值向量
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x)  # 状态价值
        advantage = self.advantage_stream(x)  # 动作优势

        # 将 V 和 A 组合成 Q 值
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value


# 定义 Double DQN
class DoubleDQN(DQN):
    """
    Double DQN 算法，通过分离动作选择和 Q 值计算，减少高估问题
    """
    def learn(self, batch_size):
        """
        更新 Q 网络，使用 Double DQN 算法

        参数：
        - batch_size: 批量大小
        """
        if len(self.memory.buffer) < batch_size:
            return

        batch = self.memory.get(batch_size)
        states = torch.tensor([d.state for d in batch], dtype=torch.float).to(self.device)
        actions = torch.tensor([d.action for d in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([d.reward for d in batch], dtype=torch.float).to(self.device)
        next_states = torch.tensor([d.next_state for d in batch], dtype=torch.float).to(self.device)
        dones = torch.tensor([d.done for d in batch], dtype=torch.float).to(self.device)

        q_values = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 使用评估网络选择动作，目标网络计算 Q 值
        next_actions = self.eval_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step_counter % 10 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
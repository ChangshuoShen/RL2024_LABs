import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections

# 超参数
EPISODES = 2000  # 训练/测试幕数
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_IETRATION = 1000  # 保存Checkpoint的间隔
MEMORY_CAPACITY = 10000  # Memory的容量
MIN_CAPACITY = 500  # 开始学习的下限
Q_NETWORK_ITERATION = 10  # 同步target network的间隔
EPSILON = 0.01  # epsilon-greedy
SEED = 0
MODEL_PATH = ''
SAVE_PATH_PREFIX = './log/dqn/'
TEST = False  # 用于控制当前行为是在训练还是在测试

# 选择一个实验环境

# Classica Control 环境 如CarrPole, MountainCar
# env = gym.make('CartPole-v1', render_mode="human" if TEST else None)
# env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# ......

# LunarLander
# env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

# 获取动作空间和状态空间的大小
NUM_ACTIONS = env.action_space.n  # 动作数量：2
NUM_STATES = env.observation_space.shape[0]  # 状态维度：4
ENV_A_SHAPE = 0 if np.issubdtype(
    type(env.action_space.sample()),
    np.integer) else env.action_space.sample().shape  # 0, 动作形状


class Model(nn.Module):
    '''Q网络模型'''
    def __init__(self, num_inputs=4):
        # TODO 输入的维度为 NUM_STATES，输出的维度为 NUM_ACTIONS
        super(Model, self).__init__()

    def forward(self, x):
        # TODO
        return x


class Data:
    '''存储单条经验的类'''
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Memory:
    """用于 Experience Replay"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 使用双端队列存储经验

    def set(self, data):
        # TODO 将数据存入经验池
        pass

    def get(self, batch_size):
        # TODO 随机采取一批数据
        pass

# 定义DQN算法的核心逻辑
class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        # 定义当前网络和目标网络
        self.eval_net, self.target_net = Model().to(device), Model().to(device)
        self.learn_step_counter = 0 # 学习步计数器
        self.memory_counter = 0 # 经验池计数器
        self.memory = Memory(capacity=MEMORY_CAPACITY) # 初始化经验池
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # 优化器
        self.loss_func = nn.MSELoss() # 损失函数

    def choose_action(self, state, EPSILON=1.0):
        ''' 根据epsilon-greedy策略选择动作 '''
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        else:
            # random policy
            action = np.random.randint(0, NUM_ACTIONS)  # int random number
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        '''将经验存储到经验池'''
        self.memory.set(data)
        self.memory_counter += 1

    def learn(self):
        '''更新Q网络'''
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            # 同步Q网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            # 保存模型参数
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # TODO 实现 Q network 的更新过程

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        '''保存当前模型'''
        torch.save(self.eval_net.state_dict(),
                   f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        '''加载模型'''
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))


def main():
    dqn = DQN()
    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}') # 初始化TensorBoard

    if TEST:
        # 测试模式,加在模型参数
        dqn.load_net(MODEL_PATH)
    for i in range(EPISODES):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED) # 重置环境并获取初始状态

        ep_reward = 0
        while True:
            # choose best action
            action = dqn.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)  
            
            # observe next state and reward
            next_state, reward, done, truncated, info = env.step(action)  
            
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            
            if TEST:
                env.render()
            if dqn.memory_counter >= MIN_CAPACITY and not TEST:
                # 经验池达到最小容量阈值的时候开始学习
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(
                        i, round(ep_reward, 3)))
            if done:
                if TEST:
                    print("episode: {} , the episode reward is {}".format(
                        i, round(ep_reward, 3)))
                break
            state = next_state
        # 每幕奖励写入Tensorboard
        writer.add_scalar('reward', ep_reward, global_step=i)


if __name__ == '__main__':
    main()

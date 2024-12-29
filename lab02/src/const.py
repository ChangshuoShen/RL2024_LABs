import torch
import os

# 超参数
EPISODES = 2000              # 训练/测试幕数
BATCH_SIZE = 64              # 每次训练的样本批量大小
LR = 0.00025                 # 学习率
GAMMA = 0.98                 # 折扣因子
SAVING_ITERATION = 1000      # 保存模型的间隔
MEMORY_CAPACITY = 10000      # 经验池的容量
MIN_CAPACITY = 500           # 开始学习的经验池最小容量
Q_NETWORK_ITERATION = 10     # 同步目标网络的间隔
EPSILON = 0.01               # epsilon-greedy 策略中的探索概率
SEED = 0                     # 随机种子
MODEL_PATH = ''              # 模型加载路径（测试模式使用）
SAVE_PATH_PREFIX = './log/dqn/'  # 模型保存路径及日志路径
TEST = False                 # 是否处于测试模式

# 初始化设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建保存路径
os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)
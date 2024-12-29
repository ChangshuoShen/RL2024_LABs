import random
import numpy as np
import torch

def set_random_seed(seed):
    """
    设置随机种子，保证实验结果可复现

    参数:
    - seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, path):
    """
    保存模型权重到指定路径

    参数:
    - model: 要保存的 PyTorch 模型
    - path: 保存路径
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """
    加载模型权重

    参数:
    - model: 要加载权重的 PyTorch 模型
    - path: 模型权重文件路径
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")


def epsilon_decay(epsilon, min_epsilon, decay_rate, episode):
    """
    epsilon 衰减函数，用于逐步减少随机探索的概率

    参数:
    - epsilon: 当前的 epsilon 值
    - min_epsilon: epsilon 的最小值
    - decay_rate: 衰减速率
    - episode: 当前的训练幕数

    返回:
    - 衰减后的 epsilon 值
    """
    new_epsilon = max(min_epsilon, epsilon * decay_rate ** episode)
    return new_epsilon
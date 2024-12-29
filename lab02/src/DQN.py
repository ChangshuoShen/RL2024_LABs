import torch
import gymnasium as gym
from src.model import DQN, Data
from src.const import *
from src.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 引入 tqdm 进度条
import numpy as np

def train_dqn(env_name, episodes, batch_size, epsilon, min_epsilon, decay_rate):
    """
    使用 DQN 算法在指定环境中训练。

    参数：
    - env_name: 环境名称（如 "CartPole-v1"）
    - episodes: 训练幕数
    - batch_size: 每次训练的样本批量大小
    - epsilon: 初始 epsilon 值（探索概率）
    - min_epsilon: epsilon 最小值
    - decay_rate: epsilon 衰减速率
    """
    # 创建环境
    env = gym.make(env_name)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # 设置随机种子
    set_random_seed(SEED)

    # 初始化 DQN 模型
    dqn = DQN(num_states, num_actions, DEVICE, LR, GAMMA, MEMORY_CAPACITY)

    # 初始化 TensorBoard Writer
    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}/{env_name}')

    # 开始训练，使用 tqdm 显示训练进度
    for episode in tqdm(range(episodes), desc=f"Training DQN on {env_name}", unit="episode"):
        state, _ = env.reset(seed=SEED)  # 重置环境，获取初始状态
        total_reward = 0  # 每幕的总奖励

        while True:
            # 根据 epsilon-greedy 策略选择动作
            action = dqn.choose_action(state, epsilon)

            # 执行动作，观察下一个状态、奖励和完成标志
            next_state, reward, done, _, _ = env.step(action)

            # 存储经验到经验池
            dqn.memory.set(Data(state, action, reward, next_state, done))

            # 累积奖励
            total_reward += reward

            # 如果经验池容量足够，则开始学习
            if len(dqn.memory.buffer) >= MIN_CAPACITY:
                dqn.learn(batch_size)

            # 如果当前幕结束，记录奖励并跳出循环
            if done:
                writer.add_scalar('Reward', total_reward, global_step=episode)
                break

            # 更新状态
            state = next_state

        # 每隔一定幕数保存模型
        if (episode + 1) % SAVING_ITERATION == 0:
            dqn_path = f"{SAVE_PATH_PREFIX}/ckpt/dqn_{env_name}_{episode + 1}.pth"
            torch.save(dqn.eval_net.state_dict(), dqn_path)
            tqdm.write(f"Saved model: {dqn_path}")  # 使用 tqdm 的写入方法避免打乱进度条

        # 更新 epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

    env.close()
    writer.close()


if __name__ == "__main__":
    # 训练 CartPole 环境
    # train_dqn(
    #     env_name="CartPole-v1",
    #     episodes=EPISODES,
    #     batch_size=BATCH_SIZE,
    #     epsilon=1.0,
    #     min_epsilon=EPSILON,
    #     decay_rate=0.995
    # )

    # # 训练 MountainCar 环境
    # train_dqn(
    #     env_name="MountainCar-v0",
    #     episodes=EPISODES,
    #     batch_size=BATCH_SIZE,
    #     epsilon=1.0,
    #     min_epsilon=EPSILON,
    #     decay_rate=0.995
    # )

    # 训练 LunarLander 环境
    train_dqn(
        env_name="LunarLander-v3",
        episodes=EPISODES,
        batch_size=BATCH_SIZE,
        epsilon=1.0,
        min_epsilon=EPSILON,
        decay_rate=0.995
    )
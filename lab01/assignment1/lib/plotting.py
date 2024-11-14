import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义一个命名元组，用于存储每个回合的长度和奖励
EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    """
    绘制山地车环境的“成本到达”函数。
    
    参数:
        env: 山地车环境实例
        estimator: 估计器，用于预测值
        num_tiles: 网格分割的数量，用于绘制
    """
    # 创建 x 和 y 的线性空间
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)  # 创建网格

    # 计算 Z 值，即对于每个 (x, y) 的最大预测值的负值
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    # 创建 3D 绘图
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)  # 设置颜色映射
    ax.set_xlabel('Position')  # 设置 x 轴标签
    ax.set_ylabel('Velocity')  # 设置 y 轴标签
    ax.set_zlabel('Value')  # 设置 z 轴标签
    ax.set_title("Mountain \"Cost To Go\" Function")  # 设置图表标题
    fig.colorbar(surf)  # 添加颜色条
    plt.show()  # 显示图形


def plot_value_function(V, title="Value Function"):
    """
    绘制值函数的表面图。
    
    参数:
        V: 值函数，映射状态到值
        title: 图表标题
    """
    # 确定 x 和 y 的范围
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    # 创建 x 和 y 的范围
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)  # 创建网格

    # 计算所有 (x, y) 坐标的值
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))  # 没有可用 A 的情况
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))  # 有可用 A 的情况

    def plot_surface(X, Y, Z, title):
        """绘制 3D 表面图的辅助函数。"""
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)  # 设置颜色映射
        ax.set_xlabel('Player Sum')  # 设置 x 轴标签
        ax.set_ylabel('Dealer Showing')  # 设置 y 轴标签
        ax.set_zlabel('Value')  # 设置 z 轴标签
        ax.set_title(title)  # 设置图表标题
        ax.view_init(ax.elev, -120)  # 设置视角
        fig.colorbar(surf)  # 添加颜色条
        plt.show()  # 显示图形

    # 绘制两种情况下的值函数
    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    """
    绘制每个回合的统计信息。
    
    参数:
        stats: EpisodeStats 实例，包含每个回合的长度和奖励
        smoothing_window: 平滑窗口大小
        noshow: 如果为 True，则不显示图表
    """
    # 绘制每个回合的长度随时间变化的图
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)  # 绘制回合长度
    plt.xlabel("Episode")  # 设置 x 轴标签
    plt.ylabel("Episode Length")  # 设置 y 轴标签
    plt.title("Episode Length over Time")  # 设置图表标题
    if noshow:
        plt.close(fig1)  # 如果 noshow 为 True，关闭图形
    else:
        plt.show(fig1)  # 显示图形

    # 绘制每个回合的奖励随时间变化的图
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)  # 绘制平滑后的奖励
    plt.xlabel("Episode")  # 设置 x 轴标签
    plt.ylabel("Episode Reward (Smoothed)")  # 设置 y 轴标签
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))  # 设置图表标题
    if noshow:
        plt.close(fig2)  # 如果 noshow 为 True，关闭图形
    else:
        plt.show(fig2)  # 显示图形

    # 绘制时间步长与回合数量的关系
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))  # 绘制累计时间步
    plt.xlabel("Time Steps")  # 设置 x 轴标签
    plt.ylabel("Episode")  # 设置 y 轴标签
    plt.title("Episode per time step")  # 设置图表标题
    if noshow:
        plt.close(fig3)  # 如果 noshow 为 True，关闭图形
    else:
        plt.show(fig3)  # 显示图形

    return fig1, fig2, fig3  # 返回所有生成的图形
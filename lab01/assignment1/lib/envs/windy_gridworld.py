import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

# 定义动作常量
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):
    """
    Windy Gridworld 环境类，继承自 gym 的离散环境。
    该环境模拟了一个带有风的网格，其中智能体需要从起点移动到目标位置，风影响智能体的移动。
    """

    metadata = {'render.modes': ['human', 'ansi']}  # 渲染模式的元数据

    def _limit_coordinates(self, coord):
        """限制坐标在环境的边界内"""
        coord[0] = min(coord[0], self.shape[0] - 1)  # 限制行坐标
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)  # 限制列坐标
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        """计算状态转移概率"""
        # 计算新的位置，考虑风的影响
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)  # 限制新位置在边界内
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)  # 将新坐标转换为状态索引
        
        # 检查是否到达目标位置 (3, 7)
        is_done = tuple(new_position) == (3, 7)
        
        # 返回转移概率，格式为 (概率, 新状态, 奖励, 是否结束)
        return [(1.0, new_state, -1.0, is_done)]  # 奖励为 -1，每一步都扣分

    def __init__(self):
        """初始化环境"""
        self.shape = (7, 10)  # 环境的形状：7 行 10 列

        nS = np.prod(self.shape)  # 状态数量
        nA = 4  # 动作数量

        # 风的强度定义
        winds = np.zeros(self.shape)  # 初始化风强度数组
        winds[:, [3, 4, 5, 8]] = 1  # 在列 3, 4, 5, 8 设置风强度为 1
        winds[:, [6, 7]] = 2  # 在列 6 和 7 设置风强度为 2

        # 计算状态转移概率
        P = {}
        for s in range(nS):  # 遍历所有状态
            position = np.unravel_index(s, self.shape)  # 将状态索引转换为坐标
            P[s] = {a: [] for a in range(nA)}  # 初始化每个状态的动作字典
            
            # 为每个动作计算转移概率
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)  # 向上
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)  # 向右
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)  # 向下
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)  # 向左

        # 设置初始状态分布：总是从 (3, 0) 开始
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0  # 起始状态概率为 1

        # 调用父类构造函数
        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        """渲染环境"""
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        """实现具体的渲染逻辑"""
        if close:
            return  # 关闭渲染时不做任何事情

        outfile = StringIO() if mode == 'ansi' else sys.stdout  # 根据模式选择输出方式

        for s in range(self.nS):  # 遍历所有状态
            position = np.unravel_index(s, self.shape)  # 获取当前状态的坐标
            # 根据当前状态决定输出内容
            if self.s == s:  # 当前状态为代理所在的状态
                output = " x "
            elif position == (3, 7):  # 到达目标状态
                output = " T "
            else:  # 普通空白状态
                output = " o "

            # 输出格式控制，去除行首和行尾的空白
            if position[1] == 0:
                output = output.lstrip() 
            if position[1] == self.shape[1] - 1:
                output = output.rstrip() 
                output += "\n"

            outfile.write(output)  # 输出当前状态的表示
        outfile.write("\n")  # 输出换行
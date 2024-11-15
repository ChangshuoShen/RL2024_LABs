import gym
import matplotlib
import numpy as np
import sys
import time
from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

#create env
env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA # 初始化动作概率
        best_action = np.argmax(Q[observation]) # 找到最佳动作
        A[best_action] += (1.0 - epsilon) # 将最佳动作的概率加上(1 - \epsilon)
        return A
    return policy_fn

def mc_first_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

#############################################Implement your code###################################################################################################
        # step 1 : Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            episode = [] # 之后用于存储(state, action, value)元组
            state = env.reset() # 重置环境，获得初始状态
            done = False
            
            while not done:
                action_probs = policy(state) # 根据当前策略选择动作概率
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs) # 根据action_probs来选择动作
                next_state, reward, done, _ = env.step(action) # 执行动作并获得下一步的状态、奖励等
                episode.append((state, action, reward)) # 存储`状态-动作-奖励`
                state = next_state
        # step 2 : Find all (state, action) pairs we've visited in this episode
            visited_pairs = set(
                (state, action) for state, action, _ in episode # 使用集合去重
            )
        # step 3 : Calculate average return for this state over all sampled episodes
        # first visit:
            G = 0
            first_visit_idx = {} # 记录一下每个状态-动作对出现的第一个位置
            first_visit_t = set()
            for t, (state, action, _) in enumerate(episode):
                if (state, action) not in first_visit_idx:
                    first_visit_idx[(state, action)] = t
                    first_visit_t.add(t)
                    
            for t, (state, action, reward) in enumerate(reversed(episode)):
                G = reward + discount_factor * G  # 计算回报
                if t in first_visit_t:
                    returns_sum[(state, action)] += G  # 累加回报
                returns_count[(state, action)] += 1  # 更新计数
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]  # 更新 Q 值
                policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)  # 更新策略
 #############################################Implement your code end###################################################################################################
    return Q, policy


def mc_every_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Step 1: Generate an episode
        episode = []
        state = env.reset()
        done = False
        
        while not done:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Step 2: Calculate returns for each (state, action) pair
        G = 0
        # Reverse iterate over the episode
        for state, action, reward in reversed(episode):
            G = reward + discount_factor * G
            returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
            
            # Update policy
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    return Q, policy

print(' #'*20, '\n', 'First-Visit Monte Carlo', '\n', ' #'*20)
for num_episodes in [
    10_000, 100_000, 500_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000]:
    
    start_time = time.time()
    Q, policy = mc_first_visit(env, num_episodes=num_episodes, epsilon=0.1)
    first_visit_time = time.time() - start_time
    print(f"\nFirst-Visit Monte Carlo (Num Episodes: {num_episodes}) took {first_visit_time:.2f} seconds.")

    # 从动作价值函数创建值函数
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions) # 选择最大动作值
        V[state] = action_value # 更新值函数
    # 绘制最佳值函数
    plotting.plot_value_function(V, title="Optimal Value Function", file_name=f"First_Visit_Value_Function_Episodes_{num_episodes}")

print(' #'*20, '\n', 'Every-Visit Monte Carlo', '\n', ' #'*20)

for num_episodes in [
    10_000, 50_000, 100_000, 500_000, 1_000_000]:
    
    start_time = time.time()
    Q, policy = mc_every_visit(env, num_episodes=num_episodes, epsilon=0.1)
    every_visit_time = time.time() - start_time
    print(f"\nEvery-Visit Monte Carlo (Num Episodes: {num_episodes}) took {every_visit_time:.2f} seconds.")

    # 从动作价值函数创建值函数
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions) # 选择最大动作值
        V[state] = action_value # 更新值函数
    # 绘制最佳值函数
    plotting.plot_value_function(V, title="Optimal Value Function",file_name=f"Every_Visit_Value_Function_Episodes_{num_episodes}")
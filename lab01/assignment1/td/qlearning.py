import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

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
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
 ########################################Implement your code here##########################################################################       
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                        
            # step 1 : Take a step
            next_state, reward, done, _ = env.step(action)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # step 2 : TD Update
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (
                reward + discount_factor * Q[next_state][best_next_action] - Q[state][action]
            )

            state = next_state
            
            if done:
                break
#######################################Implement your code end###########################################################################
    return Q, stats


def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()

        for t in itertools.count():
            action_probs = make_epsilon_greedy_policy(Q1, epsilon, env.action_space.n)(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Choose which Q value to update
            if np.random.rand() < 0.5:
                best_next_action = np.argmax(Q2[next_state])
                Q1[state][action] += alpha * (
                    reward + discount_factor * Q2[next_state][best_next_action] - Q1[state][action])
            else:
                best_next_action = np.argmax(Q1[next_state])
                Q2[state][action] += alpha * (
                    reward + discount_factor * Q1[next_state][best_next_action] - Q2[state][action])

            state = next_state
            
            if done:
                break
                
    return Q1, Q2, stats


import sarsa

def run_all_td_algs_and_compare():
    Q3, sarsa_stats = sarsa.sarsa(env, 1000)
    Q1, qlearning_stats = q_learning(env, 1000)
    Q2, Q3, double_qlearning_stats = double_q_learning(env, 1000)
    
    stats_dict = {
        'SARSA': sarsa_stats,
        'Q-Learning': qlearning_stats,
        'Double Q-Learning': double_qlearning_stats
    }

    plotting.plot_multiple_episode_stats(stats_dict)

if __name__ == '__main__':
    run_all_td_algs_and_compare()
    # print(' #'*20, '\n', 'Q-Learning', '\n', ' #'*20)
    # Q, stats = q_learning(env, 1000)

    # plotting.plot_episode_stats(stats, file_name='episode_stats_q_learning')

    # print(' #'*20, '\n', 'Double Q-Learning', '\n', ' #'*20)
    # Q1, Q2, stats =double_q_learning(env, 1000)

    # plotting.plot_episode_stats(stats, file_name='episode_stats_double_q_learning')
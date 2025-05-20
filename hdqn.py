import numpy as np
from collections import defaultdict
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from utils.replay_memory import ReplayMemory
from utils import plotting

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

# def one_hot_state(state):
#     vector = np.zeros(6)
#     vector[state-1] = 1.0
#     return np.expand_dims(vector, axis=0)

def preprocess_obs(obs):
    """处理观测元组，提取字典部分"""
    # 确保从元组中提取观测字典
    if isinstance(obs, tuple):
        obs_dict = obs[0]  # 取第一个元素（字典）
    else:
        obs_dict = obs

    # 处理图像和方向
    image = obs_dict['image'].flatten().astype(np.float32)  # 5x5x3 → 75维
    direction = np.zeros(4, dtype=np.float32)
    direction[obs_dict['direction']] = 1.0  # one-hot编码方向

    return np.concatenate([image, direction])  # 79维向量

def one_hot_goal(goal, goal_space):
    vector = np.zeros(len(goal_space))
    goal_index = goal_space.index(goal)  # 获取目标在目标空间中的位置（例如 goal=6 对应 index=0）
    vector[goal_index] = 1.0
    return vector

# def one_hot_goal(self, goal):
#     vector = np.zeros(len(self.possible_goals))
#     goal_index = self.possible_goals.index(goal)
#     vector[goal_index] = 1.0
#     return vector

def hdqn_learning(
    env,
    agent,
    num_episodes,
    exploration_schedule,
    gamma=1.0,
):
    """
    Hierarchical DQN training loop for MiniGrid environments.

    Parameters
    ----------
    env: gym.Env
        MiniGrid-compatible gym environment.
    agent:
        h-DQN agent with meta-controller and controller.
    num_episodes: int
        Total number of episodes to train for.
    exploration_schedule:
        Epsilon schedule for exploration.
    gamma: float
        Discount factor for future rewards.
    """
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )
    total_timestep = 0
    meta_timestep = 0
    ctrl_timestep = defaultdict(int)

    for i_episode in range(num_episodes):
        obs = env.reset()
        current_state = preprocess_obs(obs)  # (79,)向量

        episode_length = 0
        episode_reward = 0  # 用于统计每个 episode 的总奖励
        done = False

        while not done:
            meta_timestep += 1
            meta_epsilon = exploration_schedule.value(total_timestep)
            goal = agent.select_goal(current_state, meta_epsilon)
            encoded_goal = one_hot_goal(goal, agent.possible_goals)
            # encoded_goal = agent.one_hot_goal(goal)
            encoded_goal = encoded_goal.reshape(-1)  # shape: (6,)
            # print("encoded_goal.shape:", encoded_goal.shape)

            total_extrinsic_reward = 0
            goal_reached = False

            while not done and not goal_reached:
                total_timestep += 1
                episode_length += 1
                ctrl_timestep[goal] += 1

                ctrl_epsilon = exploration_schedule.value(total_timestep)
                joint_state_goal = np.concatenate([current_state, encoded_goal])
                # print("joint_state_goal.shape:", joint_state_goal.shape)
                action = agent.select_action(joint_state_goal, ctrl_epsilon)[0]
                print("action:", action)

                next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
                print("reward:", extrinsic_reward)

                done = terminated or truncated
                next_state = preprocess_obs(next_obs)

                # stats.episode_rewards[i_episode] += extrinsic_reward
                # stats.episode_lengths[i_episode] = episode_length

                intrinsic_reward = agent.get_intrinsic_reward(goal, next_obs)
                print("intrinsic_reward:", intrinsic_reward)
                goal_reached = agent.is_goal_reached(goal, next_obs)
                print("goal:", goal, "goal_reached:", goal_reached)

                joint_next_state_goal = np.concatenate([next_state, encoded_goal])
                agent.ctrl_replay_memory.push(
                    # joint_state_goal,
                    joint_state_goal[np.newaxis, :],
                    action,
                    # joint_next_state_goal,
                    joint_next_state_goal[np.newaxis, :],
                    intrinsic_reward,
                    done
                )

                agent.update_meta_controller(gamma)
                agent.update_controller(gamma)

                total_extrinsic_reward += extrinsic_reward
                episode_reward += extrinsic_reward
                current_state = next_state


            # After goal is completed or episode ends
            agent.meta_replay_memory.push(
                current_state, goal, next_state, total_extrinsic_reward, done
            )
        stats.episode_lengths[i_episode] = episode_length
        stats.episode_rewards[i_episode] = episode_reward

    return agent, stats

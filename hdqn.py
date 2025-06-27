import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import os
from utils import plotting
import heapq
import random
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def preprocess_obs(obs):
    if isinstance(obs, tuple):
        obs_dict = obs[0]
    else:
        obs_dict = obs

    image = obs_dict['image'].flatten().astype(np.float32)
    direction = np.zeros(4, dtype=np.float32)
    direction[obs_dict['direction']] = 1.0  # one-hot direction

    return np.concatenate([image, direction])


# def preprocess_obs(obs):
#     image = torch.from_numpy(obs['image']).float().permute(2, 0, 1)  # (3, 7, 7)
#     direction = F.one_hot(torch.tensor(obs['direction']), num_classes=4).float()  # (4,)
#     # if device:
#     #     image = image.to(device)
#     #     direction = direction.to(device)
#     return image, direction


def one_hot_goal(goal, goal_space):
    vector = np.zeros(len(goal_space))
    goal_index = goal_space.index(goal)
    vector[goal_index] = 1.0
    return vector


def hdqn_learning(
    env,
    agent,
    num_episodes,
    meta_schedule,
    ctrl_schedule,
    gamma_ctrl=0.99,
    gamma_meta=1.0,
    meta_update_freq=20,
    ctrl_update_freq=2,
):
    results_dir = r"C:\Users\Ziyi Rao\pytorch-hdqn\results"  # File save path
    top_models = []
    os.makedirs(results_dir, exist_ok=True)
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )
    total_timestep = 0
    ctrl_timestep = defaultdict(int)
    best_success_rate = 0.0

    for i_episode in range(num_episodes):
        obs,_ = env.reset()
        agent.reset_episode_flags()
        current_state = preprocess_obs(obs)
        done = False
        episode_success = False
        episode_length = 0
        episode_reward = 0
        total_extrinsic_reward = 0
        total_intrinsic_reward = 0
        while not done:
            meta_epsilon = meta_schedule.value(total_timestep)
            goal = agent.select_goal(current_state, meta_epsilon)
            print(f"[META] Selected goal: {goal}")
            encoded_goal = one_hot_goal(goal, agent.possible_goals).reshape(-1)
            goal_reached = False
            goal_start_state = current_state.copy()
            goal_extrinsic_reward = 0

            while not done and not goal_reached:
                total_timestep += 1
                episode_length += 1
                ctrl_timestep[goal] += 1
                ctrl_epsilon = ctrl_schedule.value(total_timestep)
                joint_state_goal = np.concatenate([current_state, encoded_goal])
                action = agent.select_action(joint_state_goal, ctrl_epsilon)
                next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                goal_extrinsic_reward += extrinsic_reward
                next_state = preprocess_obs(next_obs)
                goal_reached = agent.is_goal_reached(goal, next_obs)
                intrinsic_reward = agent.get_intrinsic_reward(goal, next_obs)
                joint_next_state_goal = np.concatenate([next_state, encoded_goal])
                agent.ctrl_replay_memory.push(
                    joint_state_goal[np.newaxis, :],
                    action,
                    joint_next_state_goal[np.newaxis, :],
                    intrinsic_reward,
                    done or goal_reached
                )
                if total_timestep % ctrl_update_freq == 0:
                    agent.update_controller(gamma_ctrl)

                total_extrinsic_reward += extrinsic_reward
                total_intrinsic_reward += intrinsic_reward
                episode_reward += extrinsic_reward + intrinsic_reward
                current_state = next_state

                if goal_reached:
                    print(f"Goal {goal} reached! Terminating subtask.")
                if truncated:
                    print("Episode ended due to time/step limit.")

            # After goal is completed or episode ends
            # combined_reward = total_extrinsic_reward + total_intrinsic_reward
            meta_reward = goal_extrinsic_reward
            agent.meta_replay_memory.push(goal_start_state, goal, current_state, meta_reward, done)  # total reward
            if total_timestep % meta_update_freq == 0:
                agent.update_meta_controller(gamma_meta)
            if goal == 3 and goal_reached:
                episode_success = True
        stats.episode_rewards[i_episode] += total_extrinsic_reward + total_intrinsic_reward
        stats.episode_lengths[i_episode] = episode_length
        print(f"Episode {i_episode + 1} finished:")
        print(f"  Length: {episode_length} steps")
        print(f"  Intrinsic Reward: {total_intrinsic_reward:.2f}")
        print(f"  Meta Reward: {meta_reward:.2f}")
        print(f"  Meta ε: {meta_epsilon:.3f}, Ctrl ε: {ctrl_epsilon:.3f}")
        print(f"  Success: {episode_success}")
        print("-" * 40)

        if (i_episode + 1) % 200 == 0:  # Test evey 400 episodes
            test_epsilon = 0.0  # Set exploration rate at 0
            test_stats = run_test_episodes(env, agent, num_episodes=50, epsilon=test_epsilon, render=False)
            current_success_rate = test_stats['success_rate']

            print(f"[Test] Episode {i_episode + 1} Success Rate: {current_success_rate:.2f}")
            print("-" * 40)

            # name of the file
            save_path = os.path.join(results_dir, f"model_ep{i_episode + 1}_sr{current_success_rate:.2f}.pth")

            if len(top_models) < 3 or current_success_rate > top_models[0][0]:
                torch.save({
                    'meta_controller': agent.meta_controller.state_dict(),
                    'controller': agent.controller.state_dict(),
                    'optimizer_meta': agent.meta_optimizer.state_dict(),
                    'optimizer_ctrl': agent.ctrl_optimizer.state_dict(),
                    'success_rate': current_success_rate,
                    'episode': i_episode
                }, save_path)
                print(f"Saved model to: {save_path}")

                heapq.heappush(top_models, (current_success_rate, save_path))

                # save the best 3 models
                if len(top_models) > 3:
                    worst_model = heapq.heappop(top_models)
                    if os.path.exists(worst_model[1]):
                        os.remove(worst_model[1])
                        print(f"Deleted old model: {worst_model[1]}")

            def count_layers(model):
                return sum(1 for _ in model.modules() if isinstance(_, nn.Linear))

            meta_layers = count_layers(agent.meta_controller)
            ctrl_layers = count_layers(agent.controller)
            print(f"MetaController linear layers: {meta_layers}")
            print(f"Controller linear layers: {ctrl_layers}")
            print("-" * 40)
    return agent, stats


def run_test_episodes(env, agent, num_episodes=50, epsilon=0.0, render=True):
    agent.meta_controller.eval()
    agent.controller.eval()
    success_count = 0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        current_state = preprocess_obs(obs)  # 初始状态
        done = False
        success = False
        agent.reset_episode_flags()  # 重置智能体状态

        while not done:
            # 选择目标
            # goal = agent.select_goal(current_state, epsilon)
            goal = agent.best_goal(current_state)
            encoded_goal = one_hot_goal(goal, agent.possible_goals).reshape(-1)
            goal_reached = False

            while not done and not goal_reached:
                # 选择动作
                joint_state_goal = np.concatenate([current_state, encoded_goal])
                action = agent.best_action(joint_state_goal)

                # 执行动作
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                current_state = preprocess_obs(next_obs)

                # 检查目标达成
                goal_reached = agent.is_goal_reached(goal, next_obs)
                if goal == 3 and goal_reached:
                    success = True

                if done:
                    break

        if success:
            success_count += 1
            print(f"Test Episode {ep + 1}: Success!")
        else:
            print(f"Test Episode {ep + 1}: Failed (Last goal: {goal}, Reached: {goal_reached})")
    return {"success_rate": success_count / num_episodes}

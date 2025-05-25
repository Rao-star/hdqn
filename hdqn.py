import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import os
from utils import plotting
import heapq

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):  # CUDA for GPU
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def preprocess_obs(obs):
    if isinstance(obs, tuple):
        obs_dict = obs[0]
    else:
        obs_dict = obs

    image = obs_dict['image'].flatten().astype(np.float32)
    direction = np.zeros(4, dtype=np.float32)
    direction[obs_dict['direction']] = 1.0  # one-hot direction

    return np.concatenate([image, direction])


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
    gamma=1.0,
):

    results_dir = r"C:\Users\Ziyi Rao\pytorch-hdqn\results"  # File save path
    top_models = []
    os.makedirs(results_dir, exist_ok=True)

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )
    total_timestep = 0
    meta_timestep = 0
    ctrl_timestep = defaultdict(int)

    best_success_rate = 0.0

    for i_episode in range(num_episodes):
        obs,_ = env.reset()
        agent.reset_episode_flags()
        current_state = preprocess_obs(obs)

        episode_length = 0
        episode_reward = 0
        done = False

        while not done:
            meta_timestep += 1
            meta_epsilon = meta_schedule.value(total_timestep)
            goal = agent.select_goal(current_state, meta_epsilon)
            encoded_goal = one_hot_goal(goal, agent.possible_goals)
            encoded_goal = encoded_goal.reshape(-1)

            total_extrinsic_reward = 0
            goal_reached = False

            while not done and not goal_reached:
                total_timestep += 1
                episode_length += 1
                ctrl_timestep[goal] += 1

                ctrl_epsilon = ctrl_schedule.value(total_timestep)
                joint_state_goal = np.concatenate([current_state, encoded_goal])
                action = agent.select_action(joint_state_goal, ctrl_epsilon)[0]

                next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)

                if hasattr(info, 'get') and isinstance(info, dict):
                    episode_info = info.get('episode', {})
                    extrinsic_reward = float(episode_info.get('r', extrinsic_reward))
                else:
                    extrinsic_reward = float(extrinsic_reward)

                done = terminated or truncated

                next_state = preprocess_obs(next_obs)

                intrinsic_reward = agent.get_intrinsic_reward(goal, next_obs)
                goal_reached = agent.is_goal_reached(goal, next_obs)

                joint_next_state_goal = np.concatenate([next_state, encoded_goal])
                agent.ctrl_replay_memory.push(
                    joint_state_goal[np.newaxis, :],
                    action,
                    joint_next_state_goal[np.newaxis, :],
                    intrinsic_reward,
                    done
                )

                agent.update_meta_controller(gamma)
                agent.update_controller(gamma)

                total_extrinsic_reward += extrinsic_reward
                episode_reward += extrinsic_reward
                current_state = next_state

                if goal_reached:
                    print(f"Goal {goal} reached! Terminating subtask.")
                    break

                if done:
                    print(
                        f"Episode done at total_timestep={total_timestep}, episode_length={episode_length}, info={info}")

            # After goal is completed or episode ends
            agent.meta_replay_memory.push(
                current_state, goal, next_state, total_extrinsic_reward, done
            )
        stats.episode_rewards[i_episode] += total_extrinsic_reward
        stats.episode_lengths[i_episode] = episode_length

        print(f"Episode {i_episode + 1} finished:")
        print(f"  Length: {episode_length} steps")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"Meta ε: {meta_epsilon:.3f}, Ctrl ε: {ctrl_epsilon:.3f}")
        print("-" * 40)

        if (i_episode + 1) % 200 == 0:
            test_epsilon = 0.0
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


def run_test_episodes(env, agent, num_episodes=100, epsilon=0.0, render=True):
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
            goal = agent.select_goal(current_state, epsilon)
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
            print(f"Test Episode {ep + 1}: Success=True!")
        else:
            print(f"Test Episode {ep + 1}: Failed...")
        if render:
            env.render()
    return {"success_rate": success_count / num_episodes}
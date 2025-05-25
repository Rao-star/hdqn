# test_model.py
import os
import gymnasium as gym
import numpy as np
import torch
from agents.hdqn_mdp import hDQN, OptimizerSpec
from hdqn import preprocess_obs, one_hot_goal
from minigrid.wrappers import FullyObsWrapper
from PIL import Image
# ========== Config ==========
MODEL_PATH = r"C:\Users\Ziyi Rao\pytorch-hdqn\results\try4\model_ep5400_sr0.92.pth"
ENV_NAME = "MiniGrid-DoorKey-8x8-v0"
RENDER_DELAY = 0.3
NUM_TEST_EPISODES = 5
# ============================
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def create_agent(env):
    NUM_EPISODES = 8000
    BATCH_SIZE = 256
    GAMMA = 0.95
    REPLAY_MEMORY_SIZE = 100000
    LEARNING_RATE = 0.00025

    optimizer_spec = OptimizerSpec(
        constructor=torch.optim.AdamW,
        kwargs=dict(lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.01))
    agent = hDQN(env=env,
                 optimizer_spec=optimizer_spec,
                 num_goal=4,
                 num_action=7,
                 replay_memory_size=REPLAY_MEMORY_SIZE,
                 batch_size=BATCH_SIZE)
    return agent


def load_checkpoint(agent, model_path):
    checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    agent.meta_controller.load_state_dict(checkpoint['meta_controller'])
    agent.controller.load_state_dict(checkpoint['controller'])

    agent.meta_controller.eval()
    agent.controller.eval()

    print(f"✅ Model loaded: {os.path.basename(model_path)}")
    print(f"  Trained up to episode: {checkpoint['episode']}")
    print(f"  Reported success rate: {checkpoint['success_rate']:.2f}")
    return agent


def goal_to_text(goal):
    return {
        0: "Approach key",
        1: "Pick up key",
        2: "Open door",
        3: "Reach goal"
    }.get(goal, "Unknown goal")


def action_to_text(action):
    return {
        0: "Turn left",
        1: "Turn right",
        2: "Move forward",
        3: "Pick up",
        4: "Drop",
        5: "Toggle",
        6: "Done"
    }.get(action, "Unknown action")


def visual_test(env, agent, num_episodes=5, render=True):
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        current_state = preprocess_obs(obs)
        done = False
        success = False

        agent.reset_episode_flags()

        print(f"=== Test Episode {ep + 1}/{num_episodes} ===")

        step = 0
        while not done:
            goal = agent.select_goal(current_state, epsilon=0.0)
            encoded_goal = one_hot_goal(goal, agent.possible_goals).reshape(-1)
            goal_reached = False

            while not done and not goal_reached:
                joint_state_goal = np.concatenate([current_state, encoded_goal])
                action = agent.best_action(joint_state_goal)  # 不用探索，直接选最佳动作

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                current_state = preprocess_obs(next_obs)

                goal_reached = agent.is_goal_reached(goal, next_obs)
                step += 1

                print(f"Step {step}:")
                print(f"  Goal: {goal}")
                print(f"  Action: {action}")
                print(f"  Reward: {reward}")
                print(f"  Done: {done}")
                print(f"  Goal reached: {goal_reached}")

                if render:
                    env.render()

                if done or goal_reached:
                    break

            if done:
                break

        success = info.get("success", False) or goal_reached
        print(f"Episode {ep + 1} result: {'Success' if success else 'Failure'}")
        print("=" * 30)

        if success:
            success_count += 1

    success_rate = success_count / num_episodes
    print(f"Overall success rate: {success_rate:.2f}")

    return {"success_rate": success_rate}


def visual_test_with_gif(env, agent, num_episodes=4, gif_path="test.gif"):

    frames = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        current_state = preprocess_obs(obs)
        done = False
        agent.reset_episode_flags()

        while not done:
            goal = agent.select_goal(current_state, epsilon=0.0)
            encoded_goal = one_hot_goal(goal, agent.possible_goals).reshape(-1)
            goal_reached = False

            while not done and not goal_reached:
                joint_state_goal = np.concatenate([current_state, encoded_goal])
                action = agent.best_action(joint_state_goal)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                current_state = preprocess_obs(next_obs)

                goal_reached = agent.is_goal_reached(goal, next_obs)

                frame = env.render()
                frames.append(Image.fromarray(frame))

                if done or goal_reached:
                    break
            if done:
                break

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
        )
        print(f"Saved GIF to {gif_path}")
    else:
        print("No frames captured, GIF not saved.")


if __name__ == "__main__":
    env = FullyObsWrapper(gym.make("MiniGrid-DoorKey-8x8-v0",render_mode='rgb_array'))
    try:
        print(type(env))
        print(type(env.unwrapped))
        agent = create_agent(env)
        agent = load_checkpoint(agent, MODEL_PATH)

        visual_test_with_gif(env, agent)
    finally:
        env.close()

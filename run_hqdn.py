import matplotlib.pyplot as plt
import matplotlib.style
import torch
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from agents.hdqn_mdp import hDQN, OptimizerSpec
from hdqn import hdqn_learning
from utils.plotting import plot_episode_stats
from utils.schedule import LinearSchedule
import torch.optim as optim
matplotlib.style.use('ggplot')


def main():
    NUM_EPISODES = 8000
    BATCH_SIZE = 256
    GAMMA = 0.95
    REPLAY_MEMORY_SIZE = 100000
    LEARNING_RATE = 0.00025
    ALPHA = 0.999
    EPS = 0.01

    #  AdamW Optimizer -- used in MiniGrid 8x8 DoorKey mission
    optimizer_spec = OptimizerSpec(
        constructor=torch.optim.AdamW,
        kwargs=dict(lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.01),
    )

    #  RMSprop Optimizer -- used in MiniGrid 5x5 DoorKey mission
    # optimizer_spec = OptimizerSpec(
    #     constructor=optim.RMSprop,
    #     kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    # )

    meta_schedule = LinearSchedule(
        schedule_timesteps=3_000_000, initial_p=1.0, final_p=0.05)  # 长衰减期，高探索率

    ctrl_schedule = LinearSchedule(
        schedule_timesteps=2_500_000, initial_p=1.0, final_p=0.01)  # 短衰减期，低探索率

    env = FullyObsWrapper(gym.make("MiniGrid-DoorKey-8x8-v0"))

    agent = hDQN(
        env=env,
        optimizer_spec=optimizer_spec,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=BATCH_SIZE,
    )

    # training
    agent, stats = hdqn_learning(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        meta_schedule=meta_schedule,
        ctrl_schedule=ctrl_schedule,
        gamma=GAMMA,
    )

    plot_episode_stats(stats)
    plt.show()


if __name__ == "__main__":
    main()

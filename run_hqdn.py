import minigrid
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper
from agents.hdqn_mdp import hDQN, OptimizerSpec
from hdqn import hdqn_learning
from utils.plotting import plot_episode_stats
from utils.schedule import LinearSchedule
from minigrid.envs.doorkey import DoorKeyEnv
from gymnasium.envs.registration import register
import torch.optim as optim


class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point=__name__ + ':DoorKey10x10Env'  # 使用当前模块的类
)

def main():
    NUM_EPISODES = 14_000
    BATCH_SIZE = 256
    META_GAMMA = 0.99
    CTLR_GAMMA = 0.99
    REPLAY_MEMORY_SIZE = 300_000
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
        schedule_timesteps=40_000_000, initial_p=1.0, final_p=0.01)  # longer decay period，higher exploration rate

    ctrl_schedule = LinearSchedule(
        schedule_timesteps=40_000_000, initial_p=1.0, final_p=0.1)  # shorter decay period，lower exploration rate

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = FullyObsWrapper(gym.make("MiniGrid-DoorKey-16x16-v0"))
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

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
        gamma_ctrl= CTLR_GAMMA,
        gamma_meta= META_GAMMA,
    )

    plot_episode_stats(stats)
    plt.show()


if __name__ == "__main__":
    main()

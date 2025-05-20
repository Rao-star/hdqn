import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

import torch.optim as optim
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

from envs.mdp import StochasticMDPEnv
from agents.hdqn_mdp import hDQN, OptimizerSpec
from hdqn import hdqn_learning
from utils.plotting import plot_episode_stats, plot_visited_states
from utils.schedule import LinearSchedule


def main():
    NUM_EPISODES = 100
    BATCH_SIZE = 32
    GAMMA = 1.0
    REPLAY_MEMORY_SIZE = 10000
    LEARNING_RATE = 0.001
    ALPHA = 0.99
    EPS = 0.01

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(50000, 0.1, 1)

    agent = hDQN(
        optimizer_spec=optimizer_spec,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        batch_size=BATCH_SIZE,
    )

    # 创建环境，包裹成 FullyObsWrapper 保证环境返回全观察状态
    env = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", disable_env_checker=True))

    # 训练 agent
    agent, stats,= hdqn_learning(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        exploration_schedule=exploration_schedule,
        gamma=GAMMA,
    )

    # 绘制训练过程统计图
    plot_episode_stats(stats)
    plt.show()

    # 先注释访问热力图部分，避免报错
    # plot_visited_states(visits, NUM_EPISODES)
    # plt.show()

def test_hdqn(agent, env, episodes=10):
    agent.eval()  # 切换评估模式
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # 不随机，greedy选动作
            state, reward, done, _ = env.step(action.item())
            total_reward += reward
        print(f"Test Episode {ep+1}: Reward = {total_reward}")

if __name__ == "__main__":
    main()

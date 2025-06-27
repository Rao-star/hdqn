
import random
from collections import namedtuple
from hdqn import preprocess_obs
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.replay_memory import ReplayMemory

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class MetaController(nn.Module):
    def __init__(self, in_features, out_features):
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, out_features)

    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.ln1(self.fc1(x)), 0.01)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.leaky_relu(self.ln2(self.fc2(x)), 0.01)
        x = self.fc3(x)
        return x


class MultiStrategyController(nn.Module):
    def __init__(self, in_features, out_features, num_goals=4):
        super().__init__()
        self.num_goals = num_goals

        # 共享特征提取
        self.shared_net = nn.Sequential(
            nn.Linear(in_features - num_goals, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True)  # inplace节省内存
        )

        # self.strategy_heads = nn.ModuleDict({
        #     '0': nn.Sequential(nn.Linear(256, 128), nn.Linear(128, out_features)),  # 接近钥匙策略
        #     '1': nn.Sequential(nn.Linear(256, 128), nn.Linear(128, out_features)),  # 拿钥匙策略
        #     '2': nn.Sequential(nn.Linear(256, 128), nn.Linear(128, out_features)),  # 开门策略
        #     '3': nn.Sequential(nn.Linear(256, 128), nn.Linear(128, out_features))  # 导航策略
        # })

        # 并行策略头
        self.strategy_heads = nn.ModuleList([  # 对应索引 0, 1, 2, 3
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, out_features)
            ) for _ in range(num_goals)
        ])

        # 预分配缓冲区
        self.register_buffer('_goal_mask', torch.zeros(1, dtype=torch.bool))

    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)
        state = x[:, :-self.num_goals]
        goal = x[:, -self.num_goals:]

        # 并行计算所有策略头
        shared_features = self.shared_net(state)  # [B, 256]
        all_outputs = torch.stack([head(shared_features) for head in self.strategy_heads],
                                  dim=1)  # [B, num_goals, out_features]

        # 向量化选择 (替代循环)
        goal_idx = torch.argmax(goal, dim=1)  # [B]
        return all_outputs[torch.arange(batch_size), goal_idx]  # [B, out_features]


class Controller(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        return self.net(x.float())


"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


class hDQN():
    def __init__(self,
                 env,
                 optimizer_spec,
                 num_goal=4,
                 num_action=7,
                 replay_memory_size=20_000,
                 batch_size=128):
        self.env = env
        self.num_goal = num_goal
        self.num_action = num_action
        self.batch_size = batch_size
        sample_obs, _ = env.reset()
        processed_obs = preprocess_obs(sample_obs)
        self.state_dim = len(processed_obs)
        self.controller_input_dim = self.state_dim + num_goal
        # self.meta_controller = MetaController(in_features=self.state_dim, out_features=num_goal).to(device)
        # self.controller = Controller(in_features=self.controller_input_dim, out_features=num_action).to(device)
        # self.target_meta_controller = MetaController(in_features=self.state_dim, out_features=num_goal).to(device)
        # self.target_controller = Controller(in_features=self.controller_input_dim, out_features=num_action).to(device)
        self.meta_controller = MetaController(in_features=self.state_dim, out_features=num_goal)
        self.controller = Controller(in_features=self.controller_input_dim, out_features=num_action)
        self.target_meta_controller = MetaController(in_features=self.state_dim, out_features=num_goal)
        self.target_controller = Controller(in_features=self.controller_input_dim, out_features=num_action)
        # self.controller = MultiStrategyController(in_features=self.controller_input_dim, out_features=num_action)
        # self.target_controller = MultiStrategyController(in_features=self.controller_input_dim, out_features=num_action)
        self.meta_optimizer = optimizer_spec.constructor(self.meta_controller.parameters(), **optimizer_spec.kwargs)
        self.ctrl_optimizer = optimizer_spec.constructor(self.controller.parameters(), **optimizer_spec.kwargs)
        self.meta_replay_memory = ReplayMemory(replay_memory_size)
        self.ctrl_replay_memory = ReplayMemory(replay_memory_size)
        self.completed_goals = set()
        self.possible_goals = [0, 1, 2, 3]
        self.tau = 0.01
        self._goal_checkers = {
            0: self._goal_0,
            1: self._goal_1,
            2: self._goal_2,
            3: self._goal_3
        }

    def _goal_0(self, obs):  # 靠近钥匙
        agent_pos = self.get_agent_position()
        carrying = self.env.unwrapped.carrying
        grid = self.env.unwrapped.grid
        result = ((not carrying) and any(obj.type == 'key' for obj in grid.grid if obj)
                  and self.near_key(agent_pos, obs))
        if result:
            self.completed_goals.add(0)
        return result

    def _goal_1(self, obs):  # 捡起钥匙
        carrying = self.env.unwrapped.carrying
        result = carrying is not None and carrying.type == 'key'
        if result:
            self.completed_goals.add(1)
        return result

    def _goal_2(self, obs):  # 开门
        door = next((obj for obj in self.env.unwrapped.grid.grid if obj and obj.type == 'door'), None)
        return door.is_open if door else False

    def _goal_3(self, obs):  # 到达终点
        agent_pos = self.get_agent_position()
        cell = self.env.unwrapped.grid.get(agent_pos[0], agent_pos[1])
        return cell is not None and cell.type == 'goal'

    def select_goal(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            # 利用策略
            state = torch.from_numpy(state).to(device)
            with torch.no_grad():
                q_values = self.meta_controller(state.unsqueeze(0))[0]
            return q_values.argmax().item()
        else:
            # 探索策略
            return random.randint(0, self.num_goal - 1)

    def best_goal(self, state):
        """
        用于测试：选择 Q 值最高的目标（不考虑逻辑可达性）
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.meta_controller(state_tensor)[0]
        return int(torch.argmax(q_values).item())

    def select_action(self, joint_state_goal, epsilon):
        valid_actions = [0, 1, 2, 3, 5]
        if random.random() < epsilon:
            return random.choice(valid_actions)
        # 获取 Q 值，并屏蔽无效动作
        q_values = self.controller(
            torch.tensor(joint_state_goal, dtype=torch.float32, device=device).unsqueeze(0)
        )[0]
        q_values[[4, 6]] = float('-inf')  # 将无效动作的 Q 值设为一个很小的数，让它不可能被选中
        return q_values.argmax().item()

    def best_action(self, joint_state_goal):
        """测试专用：带Q值标准化和屏蔽的贪婪策略"""
        q_values = self._compute_q_values(joint_state_goal)
        # 标准化 Q 值（视训练稳定性决定是否需要）
        q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-6)
        q_values[0, [4, 6]] = -float('inf')  # 屏蔽无效动作
        return q_values.argmax().item()

    # ===== 以下是辅助方法 =====
    def _compute_q_values(self, joint_state_goal):
        """封装Q值计算逻辑"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(joint_state_goal).float().to(device).unsqueeze(0)
            return self.controller(state_tensor)

    def get_agent_position(self):
        return self.env.unwrapped.agent_pos

    def near_key(self, agent_pos, obs):
        grid = self.env.unwrapped.grid
        width, height = self.env.unwrapped.width, self.env.unwrapped.height
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                if 0 <= x < width and 0 <= y < height:
                    cell = grid.get(x, y)
                    if cell and cell.type == 'key':
                        return True
        return False

    def is_door_open(self, obs):
        for obj in self.env.unwrapped.grid.grid:
            if obj and obj.type == 'door':
                return obj.is_open
        return False

    def at_goal_tile(self, agent_pos, obs):
        x, y = agent_pos
        if not (0 <= x < self.env.unwrapped.width and 0 <= y < self.env.unwrapped.height):
            return False
        cell = self.env.unwrapped.grid.get(x, y)
        return cell is not None and cell.type == 'goal'

    def reset_episode_flags(self):
        self.completed_goals = set()  # 每轮Episode开始时重置

    def get_intrinsic_reward(self, goal, obs):
        if goal == 3 and self.is_goal_reached(goal, obs):
            return 1.0
        elif self.is_goal_reached(goal, obs):
            return 1.0
        return 0.0

    def is_goal_reached(self, goal, obs):
        return self._goal_checkers.get(goal, lambda obs: False)(obs)

    # def is_goal_reached(self, goal, obs):
    #     agent_pos = self.get_agent_position()
    #     carrying = self.env.unwrapped.carrying
    #     grid = self.env.unwrapped.grid
    #     if goal == 0:
    #         result = ((not carrying) and any(obj.type == 'key' for obj in grid.grid if obj)
    #                   and self.near_key(agent_pos,obs))
    #         if result:
    #             self.completed_goals.add(0)  # 标记Goal 0已完成
    #         return result
    #     elif goal == 1:
    #         result = carrying is not None and carrying.type == 'key'
    #         if result:
    #             self.completed_goals.add(1)  # 标记Goal 1已完成
    #         return result
    #     elif goal == 2:  # 门已打开
    #         door = next((obj for obj in grid.grid if obj and obj.type == 'door'), None)
    #         return door.is_open if door else False
    #     elif goal == 3:  # 到达终点
    #         cell = grid.get(agent_pos[0], agent_pos[1])
    #         return cell is not None and cell.type == 'goal'
    #     return False

    @ staticmethod
    def soft_update(target_net, source_net, tau=0.01):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update_meta_controller(self, gamma=1.0):
        if len(self.meta_replay_memory) < self.batch_size:
            return
        state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask = \
            self.meta_replay_memory.sample(self.batch_size)

        # state_batch = torch.from_numpy(state_batch).to(device)
        # goal_batch = torch.from_numpy(goal_batch).long()
        # next_state_batch = torch.from_numpy(next_state_batch).to(device)
        # ex_reward_batch = torch.from_numpy(ex_reward_batch).to(device)
        # not_done_mask = torch.from_numpy(1 - done_mask).to(device)
        # goal_batch = goal_batch.to(device)
        state_batch = torch.from_numpy(state_batch)
        goal_batch = torch.from_numpy(goal_batch).long()
        next_state_batch = torch.from_numpy(next_state_batch)
        ex_reward_batch = torch.from_numpy(ex_reward_batch)
        not_done_mask = torch.from_numpy(1 - done_mask)
        goal_batch = goal_batch
        state_batch = state_batch.view(-1, self.meta_controller.fc1.in_features)
        next_state_batch = next_state_batch.view(-1, self.meta_controller.fc1.in_features)
        current_Q_values = self.meta_controller(state_batch).gather(1, goal_batch.unsqueeze(1))
        # Compute next Q value based on which goal gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = self.target_meta_controller(next_state_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = ex_reward_batch + (gamma * next_Q_values)
        # Compute Bellman error (using Huber loss)
        loss = F.smooth_l1_loss(current_Q_values, target_Q_values.unsqueeze(1))

        # Copy Q to target Q before updating parameters of Q
        # self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())
        # Optimize the model
        self.meta_optimizer.zero_grad()
        loss.backward()
        for param in self.meta_controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step()
        self.soft_update(self.target_meta_controller, self.meta_controller, self.tau)

    def update_controller(self, gamma=1.0):
        if len(self.ctrl_replay_memory) < self.batch_size:
            return
        state_goal_batch, action_batch, next_state_goal_batch, in_reward_batch, done_mask = \
            self.ctrl_replay_memory.sample(self.batch_size)
        state_goal_batch = torch.from_numpy(state_goal_batch).to(device)
        action_batch = torch.from_numpy(action_batch).long()
        next_state_goal_batch = torch.from_numpy(next_state_goal_batch).to(device)
        in_reward_batch = torch.from_numpy(in_reward_batch).to(device)
        not_done_mask = torch.from_numpy(1 - done_mask).to(device)
        action_batch = action_batch.to(device)
        # Compute current Q value, controller takes only (state, goal) and output value for every (state, goal)-action pair
        # We choose Q based on action taken.
        current_Q_values = self.controller(state_goal_batch).gather(1, action_batch.unsqueeze(1))
        # Compute next Q value based on which goal gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = self.target_controller(next_state_goal_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = in_reward_batch + (gamma * next_Q_values)
        # Compute Bellman error (using Huber loss)
        target_Q_values = target_Q_values.unsqueeze(1)
        loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

        # Copy Q to target Q before updating parameters of Q
        # self.target_controller.load_state_dict(self.controller.state_dict())
        # Optimize the model
        self.ctrl_optimizer.zero_grad()
        loss.backward()
        for param in self.controller.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.ctrl_optimizer.step()
        self.soft_update(self.target_controller, self.controller, self.tau)
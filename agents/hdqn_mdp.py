import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from utils.replay_memory import ReplayMemory, Transition

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

GOAL_TYPE = OBJECT_TO_IDX["goal"]

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


class MetaController(nn.Module):
    def __init__(self, in_features=196, out_features=4):  # 方向编码4个
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, out_features)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return self.fc3(x)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)), 0.01)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.leaky_relu(self.ln2(self.fc2(x)), 0.01)
        x = self.fc3(x)
        return x




class Controller(nn.Module):
    def __init__(self, in_features=200, out_features=7):  # 输入：网格192+方向4+子目标4   输出：7个可能动作
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(in_features,512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, out_features)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return self.fc3(x)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)), 0.01)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.leaky_relu(self.ln2(self.fc2(x)), 0.01)
        x = self.fc3(x)
        return x

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
                 replay_memory_size=20000,
                 batch_size=128):
        ###############
        # BUILD MODEL #
        ###############
        self.env = env
        self.num_goal = num_goal
        self.num_action = num_action
        self.batch_size = batch_size
        # Construct meta-controller and controller
        controller_input_dim = 192 + num_goal + 4  #(75+4 direction)
        meta_input_dim = 192 + 4   # meta-controller 只看环境状态

        self.meta_controller = MetaController(in_features=meta_input_dim, out_features=num_goal).type(dtype)
        self.target_meta_controller = MetaController(in_features=meta_input_dim, out_features=num_goal).type(dtype)
        self.controller = Controller(in_features=controller_input_dim, out_features=num_action).type(dtype)
        self.target_controller = Controller(in_features=controller_input_dim, out_features=num_action).type(dtype)
        # Construct the optimizers for meta-controller and controller
        self.meta_optimizer = optimizer_spec.constructor(self.meta_controller.parameters(), **optimizer_spec.kwargs)
        self.ctrl_optimizer = optimizer_spec.constructor(self.controller.parameters(), **optimizer_spec.kwargs)
        # Construct the replay memory for meta-controller and controller
        self.meta_replay_memory = ReplayMemory(replay_memory_size)
        self.ctrl_replay_memory = ReplayMemory(replay_memory_size)

        self.controller_update_steps = 0
        self.completed_goals = set()
        self.possible_goals = [0, 1, 2, 3]

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

    def get_intrinsic_reward(self, goal, obs):
        if goal == 3 and self.is_goal_reached(goal, obs):
            return 1.0
        # elif self.is_goal_reached(goal, obs):
        #     return 0.2
        return 0.0

    def reset_episode_flags(self):
        self.completed_goals = set()  # 每轮Episode开始时重置

    def select_goal(self, state, epsilon):
        sample = random.random()
        carrying = self.env.unwrapped.carrying
        door_open = any(obj.is_open for obj in self.env.unwrapped.grid.grid if obj and obj.type == 'door')

        # 动态生成可行目标列表（排除已完成的子目标）
        active_goals = []
        if door_open:
            active_goals = [3]
        elif carrying and carrying.type == 'key':
            active_goals = [2]
        elif any(obj.type == 'key' for obj in self.env.unwrapped.grid.grid if obj):
            possible_goals = [0, 1]
            active_goals = [g for g in possible_goals if g not in self.completed_goals]  # 关键修改
        else:
            active_goals = []

        if sample > epsilon:
            # 利用阶段：从active_goals中选择Q值最高的目标
            state = torch.from_numpy(state).type(dtype)
            with torch.no_grad():
                q_values = self.meta_controller(state.unsqueeze(0))
                if not active_goals:
                    return 3
                possible_q_values = q_values[0][active_goals]
                best_idx = torch.argmax(possible_q_values).item()
                return active_goals[best_idx]
        else:
            # 探索阶段：直接排除已完成的子目标
            return random.choice(active_goals) if active_goals else 3

    def select_action(self, joint_state_goal, epsilon):
        sample = random.random()
        if sample > epsilon:
            joint_state_goal = torch.from_numpy(joint_state_goal).type(dtype)
            with torch.no_grad():
                q_values = self.controller(joint_state_goal.unsqueeze(0))
                action = q_values.max(1)[1].cpu()
            return action
        else:
            return torch.IntTensor([random.randrange(self.num_action)])

    def best_action(self, joint_state_goal):  # 测试时用
        joint_state_goal = torch.from_numpy(joint_state_goal).type(dtype)
        with torch.no_grad():
            q_values = self.controller(joint_state_goal.unsqueeze(0))
            action = q_values.argmax(1).item()
        return action


    def is_goal_reached(self, goal, obs):
        agent_pos = self.get_agent_position()
        carrying = self.env.unwrapped.carrying
        grid = self.env.unwrapped.grid

        if goal == 0:
            result = (not carrying) and any(obj.type == 'key' for obj in grid.grid if obj) and self.near_key(agent_pos,
                                                                                                             obs)
            if result:
                self.completed_goals.add(0)  # 标记Goal 0已完成
            return result
        elif goal == 1:
            result = carrying is not None and carrying.type == 'key'
            if result:
                self.completed_goals.add(1)  # 标记Goal 1已完成
            return result
        elif goal == 2:  # 门已打开
            door = next((obj for obj in grid.grid if obj and obj.type == 'door'), None)
            return door.is_open if door else False
        elif goal == 3:  # 到达终点
            cell = grid.get(agent_pos[0], agent_pos[1])
            return cell is not None and cell.type == 'goal'
        return False

    def update_meta_controller(self, gamma=1.0):
        if len(self.meta_replay_memory) < self.batch_size:
            return
        state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask = \
            self.meta_replay_memory.sample(self.batch_size)

        state_batch = Variable(torch.from_numpy(state_batch).type(dtype))
        goal_batch = Variable(torch.from_numpy(goal_batch).long())
        next_state_batch = Variable(torch.from_numpy(next_state_batch).type(dtype))
        ex_reward_batch = Variable(torch.from_numpy(ex_reward_batch).type(dtype))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
        if USE_CUDA:
            goal_batch = goal_batch.cuda()
        # Compute current Q value, meta_controller takes only state and output value for every state-goal pair
        # We choose Q based on goal chosen.

        # reshape state_batch 变成 (batch_size, feature_dim)
        # feature_dim 就是 fc1 的输入维度，比如 79
        state_batch = state_batch.view(-1, self.meta_controller.fc1.in_features)
        next_state_batch = next_state_batch.view(-1, self.meta_controller.fc1.in_features)

        # print("state_batch.shape:", state_batch.shape)
        # print("meta_controller fc1 in_features:", self.meta_controller.fc1.in_features)
        current_Q_values = self.meta_controller(state_batch).gather(1, goal_batch.unsqueeze(1))
        # Compute next Q value based on which goal gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = self.target_meta_controller(next_state_batch).detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q
        # Compute the target of the current Q values
        target_Q_values = ex_reward_batch + (gamma * next_Q_values)
        # Compute Bellman error (using Huber loss)
        # loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
        loss = F.smooth_l1_loss(current_Q_values, target_Q_values.unsqueeze(1))

        # Copy Q to target Q before updating parameters of Q
        self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())
        # Optimize the model
        self.meta_optimizer.zero_grad()
        loss.backward()
        for param in self.meta_controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.meta_optimizer.step()

    def update_controller(self, gamma=1.0):
        if len(self.ctrl_replay_memory) < self.batch_size:
            return
        state_goal_batch, action_batch, next_state_goal_batch, in_reward_batch, done_mask = \
            self.ctrl_replay_memory.sample(self.batch_size)
        # print("state_goal_batch shape:", state_goal_batch.shape)
        state_goal_batch = Variable(torch.from_numpy(state_goal_batch).type(dtype))
        action_batch = Variable(torch.from_numpy(action_batch).long())
        next_state_goal_batch = Variable(torch.from_numpy(next_state_goal_batch).type(dtype))
        in_reward_batch = Variable(torch.from_numpy(in_reward_batch).type(dtype))
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)
        if USE_CUDA:
            action_batch = action_batch.cuda()
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
        self.target_controller.load_state_dict(self.controller.state_dict())
        # Optimize the model
        self.ctrl_optimizer.zero_grad()
        loss.backward()
        for param in self.controller.parameters():
            param.grad.data.clamp_(-1, 1)
        self.ctrl_optimizer.step()

        # self.controller_update_steps += 1
        # if self.controller_update_steps % 100 == 0:
        #     print(f"[Update {self.controller_update_steps}] Controller loss: {loss.item():.4f}")

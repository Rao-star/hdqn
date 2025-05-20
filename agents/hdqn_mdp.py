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
    def __init__(self, in_features, out_features=6):
        """
        Initialize a Meta-Controller of Hierarchical DQN network for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Controller(nn.Module):
    def __init__(self, in_features=79, out_features=7):
        """
        Initialize a Controller(given goal) of h-DQN for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class hDQN():
    """
    The Hierarchical-DQN Agent
    Parameters
    ----------
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        num_goal: int
            The number of goal that agent can choose from
        num_action: int
            The number of action that agent can choose from
        replay_memory_size: int
            How many memories to store in the replay memory.
        batch_size: int
            How many transitions to sample each time experience is replayed.
    """
    def __init__(self,
                 env,
                 optimizer_spec,
                 num_goal=4,
                 num_action=7,
                 replay_memory_size=10000,
                 batch_size=128):
        ###############
        # BUILD MODEL #
        ###############
        self.env = env
        self.num_goal = num_goal
        self.num_action = num_action
        self.batch_size = batch_size
        # Construct meta-controller and controller
        controller_input_dim = 79 + num_goal  #(75+4 direction)
        meta_input_dim = 79  # meta-controller 只看环境状态
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
        self.possible_goals = [0, 1, 2, 3]

    def get_agent_position(self):
        return self.env.unwrapped.agent_pos

    def near_key(self, agent_pos, obs):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = agent_pos[0] + dx, agent_pos[1] + dy
                cell = self.env.unwrapped.grid.get(x, y)
                if cell and cell.type == 'key':
                    return True
        return False

    def is_door_open(self, obs):
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if cell and cell.type == 'door' and cell.is_open:
                    return True
        return False

    def at_goal_tile(self, agent_pos, obs):
        x, y = agent_pos
        cell = self.env.unwrapped.grid.get(x, y)
        return cell and cell.type == 'goal'

    def get_intrinsic_reward(self, goal, obs):
        return 1.0 if self.is_goal_reached(goal, obs) else 0.0

    # def select_goal(self, state, epsilon):
    #     sample = random.random()
    #     if sample > epsilon:
    #         state = torch.from_numpy(state).type(dtype)
    #         with torch.no_grad():
    #             q_values = self.meta_controller(state.unsqueeze(0))  # 增加 batch 维度
    #             goal = q_values.max(1)[1].cpu()
    #         return goal  # 如果你想返回标量，可以改成 goal.item()
    #     else:
    #         return torch.IntTensor([random.randrange(self.num_goal)])

    def select_goal(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            state = torch.from_numpy(state).type(dtype)
            with torch.no_grad():
                q_values = self.meta_controller(state.unsqueeze(0))  # shape: [1, num_goals]
                # 只选出可能的目标中的最大 Q 值对应下标
                possible_q_values = q_values[0][self.possible_goals]  # 选取指定 index 的 q 值
                best_idx = torch.argmax(possible_q_values).item()
                goal = self.possible_goals[best_idx]
            return goal
        else:
            # 随机从可选目标中选一个
            return random.choice(self.possible_goals)

    def select_action(self, joint_state_goal, epsilon):
        sample = random.random()
        if sample > epsilon:
            joint_state_goal = torch.from_numpy(joint_state_goal).type(dtype)
            with torch.no_grad():
                q_values = self.controller(joint_state_goal.unsqueeze(0))
                action = q_values.max(1)[1].cpu()
            return action  # 同理，如需标量可用 action.item()
        else:
            return torch.IntTensor([random.randrange(self.num_action)])

    def is_goal_reached(self, goal, obs):
        agent_pos = self.get_agent_position()
        carrying = obs.get("carrying", None)

        if goal == 0:  # 走到钥匙附近
            return self.near_key(agent_pos, obs)
        elif goal == 1:  # 已拿起钥匙
            return carrying is not None and carrying['type'] == 'key'
        elif goal == 2:  # 门已打开
            return self.is_door_open(obs)
        elif goal == 3:  # 到达目标格子
            return self.at_goal_tile(agent_pos, obs)
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

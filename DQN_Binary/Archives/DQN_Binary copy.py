#######################################################################################################################
# This program will Define a Dynamic Programming algorithm to optimally rebalance a portfolio given the number of assets
# with constant mean and covariance.
#######################################################################################################################

#######################################################################################################################
# Import necessary Libraries
#######################################################################################################################

import math
import random
from collections import namedtuple, deque
from utils import *

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PortfolioRebalancerEnv import PortfolioRebalancerEnv

#######################################################################################################################
# Create Replay Memory Class
#######################################################################################################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#######################################################################################################################
# Create Q-Network
#######################################################################################################################
#
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, name, tc, chkpt_dir='results/models'):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)
        # Initialize the weights
        # nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)
        # nn.init.xavier_uniform_(self.layer3.weight)

        self.checkpoint_file = os.path.join(chkpt_dir, name + f'_tc_{tc}_assets_{n_observations}_dqn')

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class DQNTransformer(nn.Module):
    def __init__(self, n_observations, n_actions, name, tc, chkpt_dir='results/models'):
        super(DQNTransformer, self).__init__()
        self.embedding = nn.Embedding(n_observations, 64)
        self.transformer = nn.Transformer(64, 8, 2, 2, batch_first=True)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.checkpoint_file = os.path.join(chkpt_dir, name + f'_tc_{tc}_assets_{n_observations}_dqn_transformer')

    def forward(self, x):
        x = self.embedding(x.long())
        x = self.transformer(x, x)
        x = F.relu(x.mean(dim=1))  # Aggregate over the sequence dimension
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class DQNLSTM(nn.Module):
    def __init__(self, n_observations, n_actions, name, tc, chkpt_dir='results/models'):
        super(DQNLSTM, self).__init__()
        # self.embedding = nn.Embedding(n_observations, 64)
        # nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(n_observations, 64, 3, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, n_actions)

        # Initialize the weights
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)

        self.checkpoint_file = os.path.join(chkpt_dir, name + f'_tc_{tc}_assets_{n_observations}_dqn_lstm')

    def forward(self, x):
        # x = self.embedding(x.long())
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = F.relu(x.squeeze())
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


#######################################################################################################################
# Training


# RL:
# - BATCH_SIZE is the number of transitions sampled from the replay buffer
# - GAMMA is the discount factor as mentioned in the previous section
# - EPS_START is the starting value of epsilon
# - EPS_END is the final value of epsilon
# - EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# - TAU is the update rate of the target network
# - LR is the learning rate of the AdamW optimizer
#
# Finance:
# - $\mu$ is historical mean returns for each asset
#     - $\Sigma$ is the historical covariance of returns
#######################################################################################################################

class Agent(object):
    def __init__(self, mu, cov, tc, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, mem_cap=10000,
                 mode='FF'):
        # RL:
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.LR = lr
        self.mode = mode

        self.mu = mu
        self.cov = cov
        self.tc = tc
        self.w_optimal = find_optimal_wgt(mu, cov)
        self.n_assets = len(mu)

        self.env = PortfolioRebalancerEnv(
            mu=mu,
            sigma=cov,
            w_optimal=self.w_optimal,
            n_assets=self.n_assets,
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Define number of actions
        self.n_actions = self.env.action_space.shape[0]

        # Get the number of state observations
        self.state, self.info = self.env.reset()
        n_observations = self.state.shape[1]

        if self.mode == 'FF':

            self.policy_net = DQN(n_observations=n_observations, n_actions=self.n_actions,
                                  name='PolicyNet', tc=self.tc).to(self.device)
            self.target_net = DQN(n_observations=n_observations, n_actions=self.n_actions,
                                  name='TargetNet', tc=self.tc).to(self.device)
        elif self.mode == 'LSTM':
            self.policy_net = DQNLSTM(n_observations=n_observations, n_actions=self.n_actions,
                                      name='PolicyNet', tc=self.tc).to(self.device)
            self.target_net = DQNLSTM(n_observations=n_observations, n_actions=self.n_actions,
                                      name='TargetNet', tc=self.tc).to(self.device)
        elif self.mode == 'TF':
            self.policy_net = DQNTransformer(n_observations=n_observations, n_actions=self.n_actions,
                                             name='PolicyNet', tc=self.tc).to(self.device)
            self.target_net = DQNTransformer(n_observations=n_observations, n_actions=self.n_actions,
                                             name='TargetNet', tc=self.tc).to(self.device)
        else:
            raise ValueError(f"{self.mode} must be one of `FF`, `LSTM`, or `TF`!")

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.adam_optim = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(mem_cap)

        self.steps_done = 0

        self.weightA_array, self.weightA_dict = [], {}
        self.action_array, self.action_dict = [], {}
        self.reward_array, self.reward_dict = [], {}

        self.policy_net_dict = {}
        self.target_net_dict = {}
        self.dqn_result = {}

    def select_action(self, state):
        state = state[np.newaxis, :]
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        print(f"EP = {eps_threshold}")
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # optimal_action_index = policy_net(state).max(1)[1].view(1, -1)
                # output = self.policy_net(state).argmax()
                # output = torch.Tensor([output], device=self.device)
                # return output.long()
                # try:
                #     # return torch.tensor([[self.policy_net(state).argmax()]], device=self.device, dtype=torch.long)
                #     return self.policy_net(state.squeeze(0)).max(1)[1].view(1, 1)
                # except:
                #     return torch.tensor([[self.policy_net(state.squeeze(0)).argmax()]], device=self.device,
                #                         dtype=torch.int64)
                optimal_action_index = self.policy_net(state).argmax()
                output = torch.tensor(self.env.action_space[optimal_action_index],
                                      device=self.device, dtype=torch.int64).unsqueeze(0)
                return output
        else:
            # output = self.env.action_space.sample()
            # output = torch.Tensor([output], device=self.device)
            # return output.long()
            # return torch.tensor(self.env.action_space.sample(), device=self.device, dtype=torch.int64)
            temp_row_id = np.random.choice(self.env.action_space.shape[0], size=1)
            output = torch.tensor(self.env.action_space[temp_row_id], device=self.device, dtype=torch.int64)
            return output

    # def convert_action_to_index(self, action_value):
    #     return np.where(np.isclose(self.env.action_space, action_value).all(axis=1))[0]

    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # print(batch.state)
        # print(batch.action)
        # print(batch.reward)
        # print("training")

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # int_act_batch = []
        # for i in action_batch:
        #     int_act_batch.append(np.where(np.isclose(self.env.action_space, i).any(axis=1))[0].item())
        #
        # int_act_batch = torch.tensor(int_act_batch, dtype=torch.int64)[:, np.newaxis]
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        print(f"Loss = {loss.item()}")

        # Optimize the model
        self.adam_optim.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.adam_optim.step()

    def iterate(self, num_episodes):
        print("Iteration Start...\n")
        self.steps_done = 0

        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

            action = self.select_action(state)
            # print(f"Action = {action.item()}\n")
            # if action.item() not in self.env.action_space:
            #     print("HERE")
            observation, reward, done = self.env.step(action.numpy().ravel())
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)

            if done:
                next_state = None
                # store variables for plotting
                self.weightA_array.append(state.squeeze().clone().detach())
                self.reward_array.append(reward.squeeze().clone().detach())
                self.action_array.append(action)
            else:
                # get new state
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device)

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            self.state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + \
                                             target_net_state_dict[key] * (1 - self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

            # store tc results
            self.dqn_result[self.tc] = self.env

            self.weightA_dict[self.tc] = self.weightA_array
            self.action_dict[self.tc] = self.action_array
            self.reward_dict[self.tc] = self.reward_array
            self.policy_net_dict[self.tc] = self.policy_net
            self.target_net_dict[self.tc] = self.target_net

            print(f"n_asset = {self.n_assets} | Epoch {i_episode} | tc = {self.tc} " +
                  f"| Reward = {reward.squeeze().clone().detach()}"
                  )

            if i_episode % 10:
                self.save_models()

    def save_models(self):
        self.policy_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        self.policy_net.load_checkpoint()
        self.target_net.load_checkpoint()

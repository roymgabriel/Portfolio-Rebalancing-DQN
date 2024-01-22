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

import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.optimize import minimize

from PortfolioRebalancerEnv import PortfolioRebalancerEnv


#######################################################################################################################
# Define Necessary Functions
#######################################################################################################################


# Net Sharpe Ratio
def net_sharpe(w1, mu, cov, w0, tc):
    """

    :param w1: next state
    :param mu: mean
    :param cov: covariance diagonal matrix
    :param w0: current state
    :param tc: transaction costs
    :return: net sharpe value
    """
    return (w1.dot(mu) - cost_turnover(w0, w1, tc)) / np.sqrt(w1.dot(cov).dot(w1))


# Objective Function
def obj_func(x, mu, cov):
    """
    Objective Function for the Mean Variance optimization algorithm.
    :param x: tmp weight
    :param mu: mean
    :param cov: covariance diagonal matrix
    :return:
    """
    return -x.dot(mu) / np.sqrt(x.dot(cov).dot(x))
    # return 0.5 * (x.dot(cov).dot(x)) - x.dot(mu)


# Finding Optimal Weight given mean and covariance
def find_optimal_wgt(mu, cov):
    # TODO: Should we change w_max to 1 or to 2/n ? so if n = 8 limit would be 0.25 in one asset?
    n = len(mu)
    w_min = np.zeros(n)
    w_max = np.ones(n) * 2 / n
    x0 = np.ones(n) / n
    bounds = np.vstack([w_min, w_max]).T

    cstr = [{"type": "eq", "fun": lambda x: np.sum(x) - 1, "jac": lambda x: np.ones(n)}]
    opt = minimize(fun=obj_func, x0=x0, args=(mu, cov),
                   bounds=bounds,
                   constraints=cstr,
                   tol=1e-6,
                   options={"maxiter": 10000})

    if not opt.success:
        raise ValueError("optimization failed: {}".format(opt.message))

    return opt.x / opt.x.sum()


def cost_turnover(w0, w1, tc):
    """

    :param w0: current state weights
    :param w1: next state weights
    :param tc: transaction costs
    :return: cost turnover value
    """
    return np.sum(np.abs(w1 - w0) * tc) / 2


def expected_cost_total(w0, w1, opt_w, mu, cov, tc):
    """

    :param w0: current state weights
    :param w1: next state weights
    :param opt_w: optimal mean-variance weights
    :param mu: mean of returns
    :param cov: covariance of returns
    :param tc: transaction costs
    :return: expected cost of optimal - state net sharpe values
    """
    opt_net_sharpe = net_sharpe(w1=opt_w, mu=mu, cov=cov, w0=w0, tc=tc)
    w1_net_sharpe = net_sharpe(w1=w1, mu=mu, cov=cov, w0=w0, tc=tc)
    return opt_net_sharpe - w1_net_sharpe


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

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, name, tc, chkpt_dir='results/models'):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)
        self.checkpoint_file = os.path.join(chkpt_dir, name + f'_tc_{tc}_assets_{n_actions}_dqn')

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
    def __init__(self, mu, cov, tc, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr):
        # RL:
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.LR = lr

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
        # self.device = torch.device('mps:0')

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.shape[0]

        # Get the number of state observations
        self.state, self.info = self.env.reset()
        self.n_observations = self.state.shape[1]

        self.policy_net = DQN(n_observations=self.n_observations, n_actions=self.n_actions,
                              name='PolicyNet', tc=self.tc).to(self.device)
        self.target_net = DQN(n_observations=self.n_observations, n_actions=self.n_actions,
                              name='TargetNet', tc=self.tc).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.adam_optim = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(self.BATCH_SIZE)

        self.steps_done = 0

        self.weightA_array, self.weightA_dict = [], {}
        self.action_array, self.action_dict = [], {}
        self.reward_array, self.reward_dict = [], {}

        self.policy_net_dict = {}
        self.target_net_dict = {}
        self.dqn_result = {}

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # optimal_action_index = policy_net(state).max(1)[1].view(1, -1)
                optimal_action_index = self.policy_net(state).argmax()
                output = torch.tensor(self.env.action_space[optimal_action_index][np.newaxis, :], device=self.device,
                                      dtype=torch.float)
                return output
        else:
            temp_row_id = np.random.choice(self.env.action_space.shape[0], size=1)
            output = torch.tensor(self.env.action_space[temp_row_id], device=self.device, dtype=torch.float)
            return output

    # def convert_action_to_index(action_value):
    #     t = np.argsort(env.action_space)
    #     output = t[np.searchsorted(env.action_space, action_value, sorter=t)]
    #     return output

    def convert_action_to_index(self, action_value):
        return np.where(np.isclose(self.env.action_space, action_value).all(axis=1))[0]

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

        # int_act_batch = torch.tensor(np.array(list(map(convert_action_to_index, action_batch))), device=device, dtype=torch.int32)
        # int_act_batch = action_batch.apply_(convert_action_to_index).to(torch.int64)

        int_act_batch = []
        for i in action_batch:
            int_act_batch.append(np.where(np.isclose(self.env.action_space, i).any(axis=1))[0].item())

        int_act_batch = torch.tensor(int_act_batch, dtype=torch.int64)[:, np.newaxis]
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, int_act_batch)

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
        loss = criterion(state_action_values, expected_state_action_values[:, np.newaxis])
        print(f"Loss = {loss.item()}")

        # Optimize the model
        self.adam_optim.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.adam_optim.step()

    def iterate(self, num_episodes):
        print("Iteration Start...\n")

        # Get the number of state observations
        state, info = self.env.reset()

        self.steps_done = 0

        for i_episode in range(num_episodes):
            # print("Epoch {}".format(i_episode))

            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            # w_A_dict[i].append(state.squeeze().clone().detach())

            action = self.select_action(state)
            observation, reward, done = self.env.step(action.cpu().numpy().ravel())
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)

            if done:
                next_state = None
                # store variables for plotting
                self.weightA_array.append(state.squeeze().clone().detach())
                self.reward_array.append(reward.squeeze().clone().detach())
                self.action_array.append(action.squeeze().clone().detach())
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

            if i_episode % 100:
                self.save_models()

    def save_models(self):
        self.policy_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        self.policy_net.load_checkpoint()
        self.target_net.load_checkpoint()

#######################################################################################################################
# This program will Define a DQN to optimally rebalance a portfolio given the number of assets with constant mean
# and covariance.
#######################################################################################################################


#######################################################################################################################
# Import necessary Libraries
#######################################################################################################################

import numpy as np
from scipy.optimize import minimize
import random

import torch
import torch.nn as nn
import torch.optim as optim
import os


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
# Define the DQN Class

# DQN is a variant of Q-learning that uses a neural network to estimate the Q-values instead of a table.
# The neural network takes the state as input and outputs a Q-value for each action.
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#######################################################################################################################

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, n_asset, tc, chkpt_dir='results/models'):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

        self.checkpoint_file = os.path.join(chkpt_dir, f'dqn_binary_tc_{tc}_assets_{n_asset}')
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class DQNlearning(object):
    def __init__(self, mu, sigma_mat, transaction_cost, gamma, min_epsilon=0.1, learning_rate=0.001):
        # CONSTANTS
        # Define the epsilon and learning rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate

        # Define the replay buffer and batch size
        self.replay_buffer = []
        self.max_replay_buffer_size = 100000
        self.batch_size = 100

        # add loss
        self.loss = torch.tensor(0.)
        self.rewards = 0
        self.rewards_sum = 0

        # Q-NETWORK
        # Initialize the Q network and optimizer
        self.mu = mu
        self.sigma_mat = sigma_mat
        self.transaction_cost = transaction_cost
        self.gamma = gamma
        self.num_assets = len(mu)

        self.action_possible = np.array([0, 1])  # 0 or 1 : 0 no rebalancing else rebalance for 1
        self.optimal_weight = find_optimal_wgt(mu, sigma_mat)

        self.num_actions = self.action_possible.shape[0]
        # self.num_states = self.state_possible.shape[0]
        self.weights = []

        self.value_table = np.zeros(self.num_assets)
        self.q_table = np.zeros((self.num_assets, self.num_assets))  # (state x action)

        self.input_size = len(mu)
        self.output_size = self.num_actions
        self.q_network = QNetwork(self.input_size, self.output_size, tc=self.transaction_cost, n_asset=self.num_assets)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def init_state(self):
        # states from 0 to 1 and sum to 1
        state_possible = np.random.uniform(low=0, high=1, size=self.num_assets)
        state_possible /= state_possible.sum()
        state_possible = np.round(state_possible, 2)  # discretize the state space to avoid explosion

        # resample if invalid values occur in state
        while any(state_possible < 0) or any(state_possible > 1) \
                or not np.isclose(state_possible.sum(), 1.0, atol=1e-8):
            state_possible = np.random.uniform(low=0, high=1, size=self.num_assets)
            state_possible /= state_possible.sum()
            state_possible = np.round(state_possible, 2)

        assert np.isclose(state_possible.sum(), 1.0, atol=1e-8)
        # state_possible = torch.Tensor(state_possible, dtype=torch.float).to(self.device)
        return state_possible

    def get_next_state(self, state, action):
        if action == 0:
            # no rebalancing
            new_state = state
        elif action == 1:
            # TODO: Will need to change this for signal change
            new_state = self.optimal_weight
        else:
            raise ValueError("Wrong action value! Must be either 0 or 1!")
        return new_state

    def network_training_once(self, current_state):
        """
        In this code, we sample a batch of experiences from the replay buffer using the random.sample function,
        and then calculate the Q-value targets for the batch using the Q network.
        We then update the Q network using the batch by calculating the loss and calling loss.backward()
        and optimizer.step().
        Finally, we update the epsilon value using the decay factor.

        :param current_state: current_state
        :return: next_state
        """

        # Choose the action using an epsilon-greedy policy
        self.epsilon *= self.epsilon_decay
        self.epsilon = np.maximum(self.epsilon, self.min_epsilon)
        if random.uniform(0, 1) < self.epsilon:
            # exploration
            action_id = np.random.choice([0, 1])
        else:
            # exploitation
            q_values = self.q_network(
                torch.FloatTensor(current_state))  # q_table lookup now changes to NN approximation
            action_id = torch.argmax(q_values).item()

        current_action = self.action_possible[action_id]

        # Get the next state from the distribution
        next_state = self.get_next_state(current_state, current_action)

        if np.any(next_state <= 0) or np.sum(next_state) > 1:
            # bad move
            next_state = current_state
            reward = -1
        else:
            reward = -expected_cost_total(w0=current_state, w1=next_state,
                                          opt_w=self.optimal_weight, mu=self.mu, cov=self.sigma_mat,
                                          tc=self.transaction_cost)

        # Add the experience to the replay buffer
        # current_state = torch.Tensor(current_state, dtype=torch.float).to(self.device)
        # action_id = torch.Tensor(action_id, dtype=torch.int).to(self.device)
        # next_state = torch.Tensor(next_state, dtype=torch.float).to(self.device)
        # reward = torch.Tensor(reward, dtype=torch.float).to(self.device)

        self.replay_buffer.append((current_state, action_id, next_state, reward))

        # If the replay buffer is full, remove the oldest experience
        if len(self.replay_buffer) > self.max_replay_buffer_size:
            self.replay_buffer.pop(0)

        # Sample a batch of experiences from the replay buffer
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)

            # Calculate the Q-value targets for the batch using the Q network
            states = np.zeros((self.batch_size, self.input_size))
            q_targets = np.zeros((self.batch_size, self.output_size))
            reward_k_list = []
            for k in range(self.batch_size):
                # TODO: Can possibly not loop through batch_size but instead sample an entire batch immediately
                # like in CartPole

                # get batch sample
                current_state_k, action_id_k, next_state_k, reward_k = batch[k]
                # compute target Q values
                q_values_k = self.q_network(torch.FloatTensor(current_state_k))  # next Q-Value
                q_targets_this_state_all_action = q_values_k.clone().detach().numpy()
                q_targets_this_state_all_action[action_id_k] = reward_k + self.gamma * np.max(
                    q_targets_this_state_all_action)
                q_targets[k] = q_targets_this_state_all_action  # target Q-Value
                states[k] = current_state_k
                reward_k_list.append(reward_k)

            # store rewards for each batch
            self.rewards = np.mean(reward_k_list)
            self.rewards_sum = np.sum(reward_k_list)

            # Update the Q network using the batch
            self.loss = torch.tensor(0.)
            for k in range(self.batch_size):
                q_values = self.q_network(torch.FloatTensor(states[k]))
                self.loss += nn.SmoothL1Loss()(q_values, torch.FloatTensor(q_targets[k]))
            self.optimizer.zero_grad()
            self.loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
            self.optimizer.step()

        return next_state

    def iterate(self, num_episodes=1000, max_steps_per_episode=100):
        for i in range(num_episodes):
            print(
                f"Num Assets = {self.num_assets} | Epoch {i}: Loss = {self.loss.item()} | " + \
                f"Avg Batch Reward = {self.rewards} | " +
                f"Sum Batch Reward {self.rewards_sum} | epsilon = {self.epsilon}")
            current_state = self.init_state()

            if i % 100 == 0:
                self.save_models()

            for j in range(max_steps_per_episode):
                current_state = self.network_training_once(current_state=current_state)

    def save_models(self):
        self.q_network.save_checkpoint()

    def load_models(self):
        self.q_network.load_checkpoint()

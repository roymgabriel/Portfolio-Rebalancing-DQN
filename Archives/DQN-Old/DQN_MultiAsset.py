#######################################################################################################################
# This program will Define a DQN to optimally rebalance a portfolio given the number of assets with constant mean
# and covariance.
#######################################################################################################################


#######################################################################################################################
# Import necessary Libraries
#######################################################################################################################

import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
import random

import torch
import torch.nn as nn
import torch.optim as optim


#######################################################################################################################
# Define Necessary Functions
#######################################################################################################################

# Net Sharpe Ratio
def net_sharpe(w1, mu, cov, w0, tc):
    # TODO: Change this to multiasset
    return (w1.dot(mu) - cost_turnover(w0, w1, tc)) / np.sqrt(w1.dot(cov).dot(w1))


# Objective Function
def obj_func(x, mu, cov):
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
    # TODO: Change this to multiasset
    return np.sum(np.abs(w1 - w0) * tc) / 2


def expected_cost_total(w0, w1, opt_w, mu, cov, tc):
    """

    :param w0:
    :param w1:
    :param opt_w:
    :param mu:
    :param cov:
    :param tc:
    :return:
    """
    # TODO: Change this to multiasset
    opt_net_sharpe = net_sharpe(opt_w, mu, cov, w0, tc)
    w1_net_sharpe = net_sharpe(w1, mu, cov, w0, tc)
    return opt_net_sharpe - w1_net_sharpe


#######################################################################################################################
# Define the Dynamic Programming Class

# Bellman equation
# V(s) = max_a sum_s' P(s,a,s') * (r(s,a,s') + gamma * V(s'))
# where P(s,a,s') is the probability of transitioning from state s to state s' with action a,
# and gamma is a discount factor that determines the importance of future rewards.
# V(s) = max_a E_s'[ r(s,a,s') + gamma * V(s') ]
#######################################################################################################################


# class BellmanValue:
#     # it assumes fix mu and sigma_mat
#     # to consider varying mu and sigma_mat, state space must be expanded,
#     # and expectation must consider future new mu and sigma_mat
#     def __init__(self, mu, sigma_mat, transaction_cost, gamma):
#         self.mu = mu
#         self.sigma_mat = sigma_mat
#         self.transaction_cost = transaction_cost
#         self.gamma = gamma
#         x = np.arange(-99, 100)
#         self.action_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
#         self.action_possible = self.action_possible[self.action_possible.sum(axis=1) == 0, :] / 100
#         x = np.arange(1, 101)
#         self.state_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
#         self.state_possible = self.state_possible[self.state_possible.sum(axis=1) == 100, :] / 100
#         self.value_table = np.zeros(self.state_possible.shape[0])
#         self.q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))
#         self.optimal_weight = find_optimal_wgt(mu, sigma_mat)
#
#     def get_transition_prob(self, state_wgt):
#         ret_drift = self.state_possible / state_wgt
#         ret_drift -= 1
#         # this is only an approximation, it hasn't considered the weight renormalization
#         probabilities = ss.multivariate_normal.pdf(ret_drift, mean=self.mu, cov=self.sigma_mat)
#         probabilities /= np.sum(probabilities)
#         # probabilities = np.zeros(len(self.state_possible))
#         # idx = np.argmin(np.abs(self.state_possible[:, 0] - state_wgt[0]))
#         # probabilities[idx] = 1
#         return probabilities
#
#     def calculate_value(self, state_wgt):
#         action_value_current_state = []
#         for action_id in range(self.action_possible.shape[0]):
#             action = self.action_possible[action_id]
#             new_state = state_wgt + action
#             if np.any(new_state <= 0):
#                 action_value = -np.inf
#             else:
#                 transition_prob = self.get_transition_prob(new_state)
#                 reward = -expected_cost_total(state_wgt, new_state, self.optimal_weight, self.mu, self.sigma_mat,
#                                               self.transaction_cost)
#                 next_state_value = self.value_table
#                 action_value = np.sum(transition_prob * (reward + self.gamma * next_state_value))
#             action_value_current_state.append(action_value)
#         return action_value_current_state
#
#     def iterate_q_table_once(self):
#         new_value_table = np.zeros(self.state_possible.shape[0])
#         new_q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))
#         for state_id in range(self.state_possible.shape[0]):
#             state_wgt = self.state_possible[state_id]
#             new_q_table[state_id, :] = self.calculate_value(state_wgt)
#             new_value_table[state_id] = np.max(new_q_table[state_id, :])
#
#         check_converged = np.sum(np.abs(self.value_table - new_value_table))
#         self.value_table = new_value_table
#         self.q_table = new_q_table
#
#         return check_converged
#
#     def iterate(self):
#         print("Iteration Start")
#         for dummy in range(500):
#             diff = self.iterate_q_table_once()
#             if diff < 1e-5:
#                 print(f"Iteration finish at step {dummy}")
#                 break
#             print("\t iter {}: Value {}".format(dummy, diff))


#######################################################################################################################
# Define the Q-Learning Class

# Q-values
# Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
# where alpha is the learning rate (a small positive value that determines the weight given to new observations),
# r is the reward received for taking action a in state s,
# gamma is the discount factor (a value between 0 and 1 that determines the importance of future rewards),
# s' is the next state, and a' is the optimal action to take in state s' (according to the current Q-values).
# We can use an epsilon-greedy policy to select actions during the learning process. With probability epsilon,
# the agent selects a random action (exploration), and with probability 1 - epsilon, the agent selects the action
# with the highest Q-value (exploitation).
#######################################################################################################################

# class Qlearning(BellmanValue):
#     # TODO: Change self.action_possible and self.state_possible from BellmanValue Class to be sampled
#     #  instead of creating the entire mesh
#     # same assumption for constant mu and sigma_mat
#     def __init__(self, mu, sigma_mat, transaction_cost, gamma, epsilon=0.1, learning_rate=0.1):
#         super().__init__(mu, sigma_mat, transaction_cost, gamma)
#         self.epsilon = epsilon
#         self.learning_rate = learning_rate
#         self.num_actions = self.action_possible.shape[0]
#         self.num_states = self.state_possible.shape[0]
#         for state_id in range(self.num_states):
#             state = self.state_possible[state_id]
#             self.q_table[state_id, np.argwhere(np.any(state + self.action_possible <= 0, axis=1))] = -np.inf
#
#     def get_next_state(self, state, action):
#         # TODO: This has a stochastic component to it. Should we remove it? random_ret variable
#         new_state = state + action
#         # NOTE:
#         # remove stochastic component to be consistent with the value iteration because
#         # value iteration hasn't considered the weight renormalization
#         random_ret = np.random.multivariate_normal(self.mu, self.sigma_mat, size=1)
#         new_state = new_state * (1 + random_ret)
#         new_state = new_state / np.sum(new_state)
#         new_state = np.round(new_state, 2)
#         return new_state
#
#     def q_learning_once(self, state_id):
#         state_wgt = self.state_possible[state_id, :]
#
#         if random.uniform(0, 1) < self.epsilon:
#             action_feasible = np.argwhere(self.q_table[state_id, :] > -np.inf).reshape([-1])
#             action_id = np.random.choice(action_feasible, 1).item()
#         else:
#             action_id = np.argmax(self.q_table[state_id, :])
#         action = self.action_possible[action_id, :]
#
#         # Get the next state and reward
#         next_state = self.get_next_state(state_wgt, action)
#         next_state_id = np.argwhere(np.all(self.state_possible == next_state, axis=1)).item()
#         reward = -expected_cost_total(state_wgt, state_wgt + action, self.optimal_weight, self.mu, self.sigma_mat,
#                                       self.transaction_cost)
#         update_amount = reward + self.gamma * np.max(self.q_table[next_state_id, :]) - self.q_table[state_id, action_id]
#         self.q_table[state_id, action_id] += self.learning_rate * update_amount
#
#         return next_state_id
#
#     def set_value_fun_from_q_table(self):
#         for state_id in range(self.state_possible.shape[0]):
#             self.value_table[state_id] = np.max(self.q_table[state_id, :])
#
#     def iterate(self, num_episodes=1000, max_steps_per_episode=100):
#         print("Iteration Start")
#         for i in range(num_episodes):
#             print("Epoch {}".format(i))
#             current_state = random.randint(0, self.num_states - 1)
#             for j in range(max_steps_per_episode):
#                 current_state = self.q_learning_once(current_state)


#######################################################################################################################
# Define the DQN Class

# DQN is a variant of Q-learning that uses a neural network to estimate the Q-values instead of a table.
# The neural network takes the state as input and outputs a Q-value for each action.
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#######################################################################################################################

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNlearning():
    def __init__(self, mu, sigma_mat, transaction_cost, gamma, min_epsilon=0.1, learning_rate=0.001):
        # super().__init__(mu, sigma_mat, transaction_cost, gamma, min_epsilon, learning_rate)

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

        # self.state_possible = self.init_state()
        # self.action_possible = self.init_action()
        self.optimal_weight = find_optimal_wgt(mu, sigma_mat)

        # self.num_actions = self.action_possible.shape[0]
        # self.num_states = self.state_possible.shape[0]

        self.value_table = np.zeros(self.num_assets)
        self.q_table = np.zeros((self.num_assets, self.num_assets)) # state by action

        self.input_size = len(mu)
        self.output_size = self.num_assets # technically should be number of actions
        self.q_network = QNetwork(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def init_state(self):
        # states from 0 to 1 and sum to 1
        state_possible = np.random.uniform(low=0, high=1, size=self.num_assets)
        state_possible /= state_possible.sum()

        # resample if invalid values occur in state
        while any(state_possible < 0) or any(state_possible > 1):
            state_possible = np.random.uniform(low=0, high=1, size=self.num_assets)
            state_possible /= state_possible.sum()

        assert np.isclose(state_possible.sum(), 1.0, atol=1e-8)
        return state_possible

    def init_action(self, state):
        """
        TODO: We have to create all possible actions for every state, we just do not need to store all states and all
        actions. We can resample states but need to store all possible actions. Otherwise we can do DDPG or action
        exploration.
        :param state:
        :return:
        """
        # actions from -1 to 1 and sum to 0
        action_possible = np.random.uniform(low=-1, high=1, size=self.num_assets)
        action_possible = action_possible - action_possible.mean()

        tmp_q_table = state[:, np.newaxis] + action_possible  # get all possible combination of sums
        tmp_q_table = tmp_q_table.reshape(self.num_assets, self.num_assets)  # reshape to states x actions

        # resample if invalid values occur in action or combinations of action+state
        while any(action_possible < -1) or any(action_possible > 1) or any(tmp_q_table < -1) or any(tmp_q_table > 1):
            action_possible = np.random.uniform(low=-1, high=1, size=self.num_assets)
            action_possible = action_possible - action_possible.mean()

            tmp_q_table = state[:, np.newaxis] + action_possible  # get all possible combination of sums
            tmp_q_table = tmp_q_table.reshape(self.num_assets, self.num_assets)  # reshape to states x actions

        assert np.isclose(action_possible.sum(), 0.0, atol=1e-8)
        return action_possible


    def network_training_once(self, current_state, current_action):
        """
        In this code, we sample a batch of experiences from the replay buffer using the random.sample function,
        and then calculate the Q-value targets for the batch using the Q network.
        We then update the Q network using the batch by calculating the loss and calling loss.backward()
        and optimizer.step().
        Finally, we update the epsilon value using the decay factor.

        :param current_state: current_state
        :return: next_state
        """
        # TODO: Need to sample this possible state
        # state_wgt = self.state_possible[state_id, :]


        # Choose the action using an epsilon-greedy policy
        self.epsilon *= self.epsilon_decay
        self.epsilon = np.maximum(self.epsilon, self.min_epsilon)
        if random.uniform(0, 1) < self.epsilon:
            # action_feasible = np.argwhere(self.q_table[state_id, :] > -np.inf).reshape([-1])
            # action_id = np.random.choice(action_feasible, 1).item()

            # resample a current action to avoid exploitation
            current_action = self.init_action(state=current_state)  # aka action delta
        else:
            q_values = self.q_network(torch.FloatTensor(current_state))  # q_table lookup now changes to NN approximation
            action_id = torch.argmax(q_values).item()
            # action_delta =

        action_delta = self.action_possible[action_id]

        # Get the distribution of next states and rewards for the current state and action
        # reward_dist = rewards[next_state_dist]

        # Get the next state from the distribution
        next_state = self.get_next_state(state_wgt, action_delta)

        if np.any(next_state <= 0):
            next_state_id = state_id
            print("HERE")
            reward = -1
        else:
            next_state_id = np.argwhere(np.all(self.state_possible == next_state, axis=1)).item()
            reward = -expected_cost_total(state_wgt, state_wgt + action_delta, self.optimal_weight,
                                          self.mu, self.sigma_mat, self.transaction_cost)

        # Add the experience to the replay buffer
        self.replay_buffer.append((state_id, action_id, next_state_id, reward))

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
                state_id_k, action_id_k, next_state_id_k, reward_k = batch[k]
                # get possible states at batch k
                state_wgt_k = self.state_possible[state_id_k]
                # compute target Q values
                q_values_k = self.q_network(torch.FloatTensor(state_wgt_k))  # next Q-Value
                q_targets_this_state_all_action = q_values_k.clone().detach().numpy()
                q_targets_this_state_all_action[action_id_k] = reward_k + self.gamma * np.max(
                    q_targets_this_state_all_action)
                q_targets[k] = q_targets_this_state_all_action  # target Q-Value
                states[k] = state_wgt_k
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

        return next_state_id

    def iterate(self, num_episodes=1000, max_steps_per_episode=100):
        for i in range(num_episodes):
            print(
                f"Epoch {i}: Loss = {self.loss.item()} | Avg Batch Reward = {self.rewards} | Sum Batch Reward {self.rewards_sum} | epsilon = {self.epsilon}")
            # current_state = random.randint(0, self.num_states - 1)
            current_state = self.init_state()

            for j in range(max_steps_per_episode):
                current_state = self.network_training_once(current_state=current_state)
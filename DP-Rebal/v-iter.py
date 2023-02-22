import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
import scipy
import random


def cost_turnover(cost_vec, wgt_orig, wgt_target):
    return np.sum(np.abs(wgt_orig - wgt_target) * cost_vec)


def certainty_equivalent_ret(wgt, mu, sigma_mat):
    return np.sum(wgt * mu) - 0.5 * np.dot(wgt, np.dot(sigma_mat, wgt))


def solve_quadratic_optimization_st_linear_constraint(mu, sigma_mat):
    # Define the objective function and its gradient
    def obj_func(x, Q, c):
        return 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c, x)

    def obj_grad(x, Q, c):
        return np.dot(Q, x) + c

    # Define the constraint functions and their gradients
    def linear_constraint(x, A, b):
        return np.dot(A, x) - b

    def linear_constraint_jac(x, A, b):
        return A

    # Define the problem parameters
    Q = sigma_mat
    c = -mu
    A = np.ones_like(mu)
    b = np.array([1])

    # Define the bounds on x
    bounds = [(0, 1)] * len(mu)

    # Define the constraints
    linear_constraints = {'type': 'eq', 'fun': linear_constraint, 'jac': linear_constraint_jac, 'args': (A, b)}

    # Solve the problem
    x0 = np.ones_like(mu) / len(mu)
    result = minimize(obj_func, x0=x0, args=(Q, c), jac=obj_grad, bounds=bounds, constraints=[linear_constraints])
    return result


def cost_suboptimality(wgt_current, mu, sigma_mat):
    utility_optimal = solve_quadratic_optimization_st_linear_constraint(mu, sigma_mat)
    np.testing.assert_almost_equal(-utility_optimal.fun, certainty_equivalent_ret(utility_optimal.x, mu, sigma_mat))
    ret_ce_optimal = certainty_equivalent_ret(utility_optimal.x, mu, sigma_mat)
    ret_ce_current = certainty_equivalent_ret(wgt_current, mu, sigma_mat)
    return ret_ce_optimal - ret_ce_current


def expected_cost_total(state_wgt, action_delta, mu, sigma_mat, transaction_cost):
    cost_p1 = cost_turnover(transaction_cost, state_wgt, state_wgt + action_delta)
    cost_p2 = cost_suboptimality(state_wgt + action_delta, mu, sigma_mat)
    return cost_p1 + cost_p2


# two-asset model with dynamic programing and q-table learning

# Bellman equation
# V(s) = max_a sum_s' P(s,a,s') * (r(s,a,s') + gamma * V(s'))
# where P(s,a,s') is the probability of transitioning from state s to state s' with action a, and gamma is a discount factor that determines the importance of future rewards.
# V(s) = max_a E_s'[ r(s,a,s') + gamma * V(s') ]
class BellmanValue:
    # it assumes fix mu and sigma_mat
    # to consider varying mu and sigma_mat, state space must be expanded, and expectation must consider future new mu and sigma_mat
    def __init__(self, mu, sigma_mat, transaction_cost, gamma):
        self.mu = mu
        self.sigma_mat = sigma_mat
        self.transaction_cost = transaction_cost
        self.gamma = gamma
        x = np.arange(-7, 8)
        self.action_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        self.action_possible = self.action_possible[self.action_possible.sum(axis=1) == 0, :]
        x = np.arange(1, 101)
        self.state_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        self.state_possible = self.state_possible[self.state_possible.sum(axis=1) == 100, :]
        self.value_table = np.zeros(self.state_possible.shape[0])
        self.q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))

    def get_transition_prob(self, state_current, action):
        state_new_wgt = state_current + action
        ret_drift = self.state_possible / state_new_wgt
        ret_drift -= 1
        # this is only an approximation, it hasn't considered the weight renormalization
        probabilities = scipy.stats.multivariate_normal.pdf(ret_drift, mean=self.mu, cov=self.sigma_mat)
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_value(self, state_wgt):
        action_value_current_state = []
        for action_id in range(self.action_possible.shape[0]):
            action = self.action_possible[action_id]
            new_state = state_wgt + action
            if np.any(new_state <= 0):
                action_value = -np.inf
            else:
                transition_prob = self.get_transition_prob(state_wgt, action)
                reward = -expected_cost_total(state_wgt/100, action/100, self.mu, self.sigma_mat, self.transaction_cost)
                next_state_value = self.value_table
                action_value = np.sum(transition_prob * (reward + self.gamma * next_state_value))
            action_value_current_state.append(action_value)
        return action_value_current_state

    def iterate_q_table_once(self):
        new_value_table = np.zeros(self.state_possible.shape[0])
        new_q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))
        for state_id in range(self.state_possible.shape[0]):
            state_wgt = self.state_possible[state_id]
            new_q_table[state_id, :] = self.calculate_value(state_wgt)
            new_value_table[state_id] = np.max(new_q_table[state_id, :])

        check_converged = np.sum(np.abs(self.value_table - new_value_table))
        self.value_table = new_value_table
        self.q_table = new_q_table

        return check_converged


mu = np.array([50, 200]) / 1e4
sigma = np.array([300, 800]) / 1e4
cov = np.diag(sigma ** 2)
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2019, 12, 31)
dates = pd.date_range(start, end, freq="M")
ret = np.random.multivariate_normal(mu / 12, cov / 12, size=len(dates))
ret_df = pd.DataFrame(ret, index=dates)
trans_cost = 10/1e4

self = bell = BellmanValue(mu, cov, trans_cost, gamma=0.9)
for dummy in range(200):
    diff = bell.iterate_q_table_once()
    if diff < 1e-4:
        break
    print("Iter {}: Value {}".format(dummy, diff))


# Q-values
# Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
# where alpha is the learning rate (a small positive value that determines the weight given to new observations), r is the reward received for taking action a in state s,
# gamma is the discount factor (a value between 0 and 1 that determines the importance of future rewards), s' is the next state, and a' is the optimal action to take in state s' (according to the current Q-values).
# We can use an epsilon-greedy policy to select actions during the learning process. With probability epsilon, the agent selects a random action (exploration), and with probability 1 - epsilon, the agent selects the action with the highest Q-value (exploitation).
class Qlearning(BellmanValue):
    # same assumption for constant mu and sigma_mat
    def __init__(self, mu, sigma_mat, transaction_cost, gamma, epsilon=0.1, learning_rate=0.1):
        super().__init__(mu, sigma_mat, transaction_cost, gamma)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_actions = self.action_possible.shape[0]
        self.num_states = self.state_possible.shape[0]
        for state_id in range(self.num_states):
            state = self.state_possible[state_id]
            self.q_table[state_id, np.argwhere(np.any(state + self.action_possible <= 0, axis=1))] = -np.inf

    def get_next_state(self, state, action):
        new_state = state + action
        # remove stochastic component to be consistent with the value iteration because value iteration hasn't considered the weight renormalization
        # random_ret = np.random.multivariate_normal(self.mu, self.sigma_mat, size=1)
        # new_state = new_state * (1+random_ret)
        # new_state = new_state / np.sum(new_state) * 100
        # new_state = np.round(new_state)
        # new_state = (new_state / np.sum(new_state) * 100)
        # new_state = np.round(new_state).astype(int)
        return new_state

    def q_learning_once(self, state_id):
        state_wgt = self.state_possible[state_id, :]

        if random.uniform(0, 1) < self.epsilon:
            action_feasible = np.argwhere(self.q_table[state_id, :] > -np.inf).reshape([-1])
            action_id = np.random.choice(action_feasible, 1).item()
        else:
            action_id = np.argmax(self.q_table[state_id, :])
        action = self.action_possible[action_id, :]

        # Get the next state and reward
        next_state = self.get_next_state(state_wgt, action)
        next_state_id = np.argwhere(np.all(self.state_possible == next_state, axis=1)).item()
        reward = -expected_cost_total(state_wgt/100, action/100, self.mu, self.sigma_mat, self.transaction_cost)
        update_amount = reward + self.gamma * np.max(self.q_table[next_state_id, :]) - self.q_table[state_id, action_id]
        self.q_table[state_id, action_id] += self.learning_rate * update_amount

        return next_state_id

    def set_value_fun_from_q_table(self):
        for state_id in range(self.state_possible.shape[0]):
            self.value_table[state_id] = np.max(self.q_table[state_id, :])


self = qlearner = Qlearning(mu, cov, trans_cost, gamma=0.9, epsilon=0.1, learning_rate=0.1)

num_episodes = 1000
max_steps_per_episode = 100

for i in range(num_episodes):
    print("Epoch {}".format(i))
    current_state = random.randint(0, qlearner.num_states - 1)
    for j in range(max_steps_per_episode):
        current_state = qlearner.q_learning_once(current_state)

qlearner.q_table
qlearner.set_value_fun_from_q_table()
qlearner.value_table
bell.value_table

from matplotlib import pyplot as plt
plt.plot(bell.value_table)
plt.show()

plt.plot(qlearner.value_table)
plt.show()


# DQN is a variant of Q-learning that uses a neural network to estimate the Q-values instead of a table.
# The neural network takes the state as input and outputs a Q-value for each action.
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import torch.nn as nn
import torch.optim as optim

# Define the Q network
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

class DQNlearning(Qlearning):
    def __init__(self, mu, sigma_mat, transaction_cost, gamma, epsilon=0.1, learning_rate=0.001):
        super().__init__(mu, sigma_mat, transaction_cost, gamma, epsilon, learning_rate)

        # Initialize the Q network and optimizer
        self.input_size = len(mu)
        self.output_size = self.num_actions
        self.q_network = QNetwork(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Define the epsilon and learning rate
        self.epsilon = 1.0
        self.min_epsilon = epsilon
        self.epsilon_decay = 0.999
        self.learning_rate = learning_rate

        # Define the replay buffer and batch size
        self.replay_buffer = []
        self.max_replay_buffer_size = 100000
        self.batch_size = 32

    def network_training_once(self, state_id):
        input_size = self.num_states
        output_size = self.num_actions

        state_wgt = self.state_possible[state_id, :]

        # Choose the action using an epsilon-greedy policy
        self.epsilon *= self.epsilon_decay
        self.epsilon = np.maximum(self.epsilon, self.min_epsilon)
        if random.uniform(0, 1) < self.epsilon:
            action_feasible = np.argwhere(self.q_table[state_id, :] > -np.inf).reshape([-1])
            action_id = np.random.choice(action_feasible, 1).item()
        else:
            q_values = self.q_network(torch.FloatTensor(state_wgt))  # q_table lookup now changes to NN approximation
            action_id = torch.argmax(q_values).item()

        action_delta = self.action_possible[action_id]

        # Get the distribution of next states and rewards for the current state and action
        # reward_dist = rewards[next_state_dist]

        # Get the next state from the distribution
        next_state = self.get_next_state(state_wgt, action_delta)

        if np.any(next_state <= 0):
            next_state_id = state_id
            reward = -1
        else:
            next_state_id = np.argwhere(np.all(self.state_possible == next_state, axis=1)).item()
            reward = -expected_cost_total(state_wgt / 100, action_delta / 100, self.mu, self.sigma_mat, self.transaction_cost)

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
            for k in range(self.batch_size):
                state_id_k, action_id_k, next_state_id_k, reward_k = batch[k]
                state_wgt_k = self.state_possible[state_id_k]
                q_values_k = self.q_network(torch.FloatTensor(state_wgt_k))
                q_targets_this_state_all_action = q_values_k.clone().detach().numpy()
                q_targets_this_state_all_action[action_id_k] = reward_k + self.gamma * np.max(self.q_network(torch.FloatTensor(state_wgt_k)).detach().numpy())
                q_targets[k] = q_targets_this_state_all_action
                states[k] = state_wgt_k

            # Update the Q network using the batch
            self.optimizer.zero_grad()
            loss = torch.tensor(0.)
            for k in range(self.batch_size):
                q_values = self.q_network(torch.FloatTensor(states[k]))
                loss += nn.MSELoss()(q_values, torch.FloatTensor(q_targets[k]))
            loss.backward()
            self.optimizer.step()

        # In this code, we sample a batch of experiences from the replay buffer using the random.sample function,
        # and then calculate the Q-value targets for the batch using the Q network.
        # We then update the Q network using the batch by calculating the loss and calling loss.backward() and optimizer.step().
        # Finally, we update the epsilon value using the decay factor.
        return next_state_id

self = dqn = DQNlearning(mu, cov, trans_cost, gamma=0.9, epsilon=0.1, learning_rate=0.001)

for i in range(num_episodes):
    print("Epoch {}".format(i))
    current_state = random.randint(0, qlearner.num_states - 1)
    for j in range(max_steps_per_episode):
        current_state = dqn.network_training_once(current_state)

import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
import scipy
import random

import torch
import torch.nn as nn
import torch.optim as optim


def reward_sharpe_net_tc(w0, d1, mu, cov, tc):
    w2 = w0 + d1

    # optimal net sharpe
    net_sharpe2 = (w2.dot(mu) - np.sum(np.abs(d1) * tc)) / np.sqrt(w2.dot(cov).dot(w2))

    return net_sharpe2


def find_optimal_wgt(w0, mu, cov, tc):
    # Define the objective function to be optimized
    def objective(d1):
        return -reward_sharpe_net_tc(w0, d1, mu, cov, tc)

    # Define the constraint function
    def constraint(d1):
        return d1.sum()

    # Define the bounds for d1
    d1_init = np.zeros(len(mu))
    bounds = [(-w0[_], None) for _ in range(len(d1_init))]

    # Define the optimization problem
    problem = {
        'fun': objective,
        'x0': d1_init,
        'bounds': bounds,
        'constraints': [{'type': 'eq', 'fun': constraint}]
    }

    # Solve the optimization problem
    result = minimize(**problem)

    # Extract the optimized value of d1
    return result.x


def cost_suboptimality(w0, d1, mu, cov, tc):
    # to calculate net sharpe difference computational cose will be too high -- optimization is needed to solve for the current optimum
    d_opt = find_optimal_wgt(w0, mu, cov, tc)
    reward_optimal = reward_sharpe_net_tc(w0, d_opt, mu, cov, tc)
    reward_current = reward_sharpe_net_tc(w0, d1, mu, cov, tc)
    return reward_optimal - reward_current


# two-asset model with dynamic programing and q-table learning

# Bellman equation
# V(s) = max_a sum_s' P(s,a,s') * (r(s,a,s') + gamma * V(s'))
# where P(s,a,s') is the probability of transitioning from state s to state s' with action a, and gamma is a discount factor that determines the importance of future rewards.
# V(s) = max_a E_s'[ r(s,a,s') + gamma * V(s') ]
class BellmanValue:
    # mu will change from period to period through mu_change_cov, which can be set through one-year forward PTVA study
    def __init__(self, num_asset, sigma_mat, mu_change_cov, transaction_cost, gamma):
        self.sigma_mat = sigma_mat
        self.mu_change_cov = mu_change_cov
        self.transaction_cost = transaction_cost
        self.gamma = gamma
        self.num_asset = num_asset
        x = np.arange(-7, 8)
        self.action_possible = np.array(np.meshgrid(*([x] * self.num_asset))).T.reshape(-1, self.num_asset)
        self.action_possible = self.action_possible[self.action_possible.sum(axis=1) == 0, :]
        x1 = np.arange(1, 101)
        x2 = np.arange(0, 401, 5)
        self.state_possible = np.array(np.meshgrid(*([x1] * self.num_asset + [x2] * self.num_asset))).T.reshape(-1, 2*self.num_asset)
        self.state_possible = self.state_possible[self.state_possible[:, 0:self.num_asset].sum(axis=1) == 100, :]
        self.state_col_wgt = range(0, self.num_asset)
        self.state_col_mu = range(self.num_asset, 2*self.num_asset)
        self.value_table = np.zeros(self.state_possible.shape[0])
        self.q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))

    def get_transition_prob(self, state_current, action):
        state_new_wgt = state_current[self.state_col_wgt] + action
        ret_drift = self.state_possible[:, self.state_col_wgt] / state_new_wgt
        ret_drift -= 1
        # this is only an approximation, it hasn't considered the weight renormalization
        prob_on_wgt = scipy.stats.multivariate_normal.logpdf(ret_drift, mean=state_current[self.state_col_mu]/10000, cov=self.sigma_mat)
        prob_on_wgt -= prob_on_wgt.max()

        mu_change = self.state_possible[:, self.state_col_mu] - state_current[self.state_col_mu]
        x = np.arange(-400, 401, 5)
        mu_allow_neg = np.array(np.meshgrid(*([x] * self.num_asset))).T.reshape(-1, self.num_asset)
        prob_on_mu_change = scipy.stats.multivariate_normal.logpdf(mu_allow_neg, mean=state_current[self.state_col_mu], cov=self.mu_change_cov)
        df = pd.DataFrame(mu_allow_neg)
        df['prob'] = prob_on_mu_change
        mu_no_neg = np.maximum(mu_allow_neg, 0)
        df.iloc[:, :self.num_asset] = mu_no_neg
        df = df.groupby(list(range(self.num_asset))).sum().reset_index()
        prob_on_mu_change = df['prob'].values
        prob_on_mu_change -= prob_on_mu_change.max()
        prob_on_mu_change = [x for x in prob_on_mu_change for _ in range(self.state_possible.shape[0] // len(prob_on_mu_change)) ]

        probabilities = prob_on_wgt + np.array(prob_on_mu_change)
        probabilities -= probabilities.max()
        probabilities = np.exp(probabilities)
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_value(self, state_wgt):
        action_value_current_state = []
        for action_id in range(self.action_possible.shape[0]):
            action = self.action_possible[action_id]
            new_state = state_wgt[self.state_col_wgt] + action
            if np.any(new_state <= 0):
                action_value = -np.inf
            else:
                transition_prob = self.get_transition_prob(state_wgt, action)
                reward = reward_sharpe_net_tc(state_wgt[self.state_col_wgt]/100, action/100, state_wgt[self.state_col_mu]/10000, self.sigma_mat, self.transaction_cost)
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


class DQNlearning(BellmanValue):
    # TODO state_id is actually not needed, use state_wgt_mu directly
    def __init__(self, num_asset, sigma_mat, mu_change_cov, transaction_cost, gamma, epsilon=0.1, learning_rate=0.001, mu_error=0):
        super().__init__(num_asset, sigma_mat, mu_change_cov, transaction_cost, gamma)
        self.mu_error = mu_error

        # Initialize the Q network and optimizer
        self.input_size = self.num_asset * 2
        self.num_actions = self.action_possible.shape[0]
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

        self.num_states = self.state_possible.shape[0]
        self.action_feasible = list()
        for state_id in range(self.num_states):
            state = self.state_possible[state_id]
            self.action_feasible.append(
                np.argwhere(np.all(state[self.state_col_wgt] + self.action_possible > 0, axis=1)).reshape([-1])
            )

    # mean 0, delta sd 30, error 50 - 100

    def get_next_state(self, state, action):
        new_state_wgt = state[self.state_col_wgt] + action
        mu = state[self.state_col_mu] / 1e4
        random_ret = np.random.multivariate_normal(mu, self.sigma_mat, size=1)
        random_ret += np.random.normal(0, self.mu_error / 1e4, size=len(random_ret))
        new_state_wgt = new_state_wgt * (1+random_ret)
        new_state_wgt = new_state_wgt / np.sum(new_state_wgt) * 100
        new_state_wgt = np.maximum(new_state_wgt, 1)
        new_state_wgt = np.round(new_state_wgt)
        new_state_wgt = (new_state_wgt / np.sum(new_state_wgt) * 100)
        new_state_wgt = np.round(new_state_wgt).astype(int)

        new_state_mu = np.random.multivariate_normal(mu, self.mu_change_cov, size=1)
        new_state_mu = np.round(new_state_mu/5) * 5
        new_state_mu = np.maximum(new_state_mu, 0)
        new_state_mu = np.minimum(new_state_mu, 400)
        new_state = np.concatenate((new_state_wgt.reshape([-1]), new_state_mu.reshape([-1])), axis=0)
        new_state = np.round(new_state).astype(int)
        return new_state


    def find_best_action(self, state_wgt_mu):
        q_values = self.q_network(torch.FloatTensor(state_wgt_mu))  # q_table lookup now changes to NN approximation
        action_id = torch.argmax(q_values).item()
        action_delta = self.action_possible[action_id]
        return dict(action_id=action_id, wgt_delta=action_delta)


    def network_training_once(self, state_id):
        input_size = self.num_states
        output_size = self.num_actions

        state_wgt = self.state_possible[state_id, :]

        # Choose the action using an epsilon-greedy policy
        self.epsilon *= self.epsilon_decay
        self.epsilon = np.maximum(self.epsilon, self.min_epsilon)
        if random.uniform(0, 1) < self.epsilon:
            action_feasible = self.action_feasible[state_id]
            action_id = np.random.choice(action_feasible, 1).item()
        else:
            q_values = self.q_network(torch.FloatTensor(state_wgt))  # q_table lookup now changes to NN approximation
            action_id = torch.argmax(q_values).item()

        action_delta = self.action_possible[action_id]

        # Get the distribution of next states and rewards for the current state and action
        # reward_dist = rewards[next_state_dist]

        # Get the next state from the distribution
        next_state = self.get_next_state(state_wgt, action_delta)

        if np.any(next_state < 0):
            next_state_id = state_id
            reward = -1
        else:
            next_state_id = np.argwhere(np.all(self.state_possible == next_state, axis=1)).item()
            # another idea: use the realized return for reward, but it may not work because return could be too noisy
            # reward = -cost_suboptimality(state_wgt[self.state_col_wgt]/100, action_delta/100, state_wgt[self.state_col_mu]/10000, self.sigma_mat, self.transaction_cost)
            reward = reward_sharpe_net_tc(state_wgt[self.state_col_wgt]/100, action_delta/100, state_wgt[self.state_col_mu]/10000, self.sigma_mat, self.transaction_cost)

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


if __name__ == '__main__':

    num_asset = 2
    mu = np.array([50, 200]) / 1e4
    sigma = np.array([300, 800]) / 1e4
    cov = np.diag(sigma ** 2)
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2019, 12, 31)
    dates = pd.date_range(start, end, freq="M")
    ret = np.random.multivariate_normal(mu / 12, cov / 12, size=len(dates))
    ret_df = pd.DataFrame(ret, index=dates)
    trans_cost = 10 / 1e4
    pvta_sd = np.array([50, 50])
    mu_change_cov = np.diag(pvta_sd ** 2)

    # self = bell = BellmanValue(mu, cov, mu_change_cov, trans_cost, gamma=0.9)
    # for dummy in range(200):
    #     diff = bell.iterate_q_table_once()
    #     if diff < 1e-4:
    #         break
    #     print("Iter {}: Value {}".format(dummy, diff))

    self = dqn = DQNlearning(num_asset, cov, mu_change_cov, trans_cost, gamma=0.9, epsilon=0.1, learning_rate=0.001)
    self.get_next_state(np.array([43, 57, 3, 3]), np.array([-1,1]))

    num_episodes = 1000
    max_steps_per_episode = 100

    for i in range(num_episodes):
        print("Epoch {}".format(i))
        current_state = random.randint(0, dqn.num_states - 1)
        for j in range(max_steps_per_episode):
            current_state = dqn.network_training_once(current_state)

    best_action = dqn.value_table.copy()
    for state_id in range(dqn.num_states):
        q_values = dqn.q_network(torch.FloatTensor(dqn.state_possible[state_id]))
        dqn.value_table[state_id] = q_values.max().detach().numpy()
        best_action[state_id] = torch.argmax(q_values).detach().numpy()

    from matplotlib import pyplot as plt
    plt.plot(dqn.value_table)
    plt.show()
    plt.savefig('figdqn1.png')
    dqn.state_possible[:99,:]

    plt.plot(dqn.value_table[:99])
    plt.show()

    chunk_size = 99
    i = 900
    plt.plot(dqn.value_table[(i-1)*chunk_size:i*chunk_size])
    print(dqn.state_possible[(i-1)*chunk_size:i*chunk_size, :])
    plt.show()

    plt.plot(best_action[(i-1)*chunk_size:i*chunk_size])
    plt.show()


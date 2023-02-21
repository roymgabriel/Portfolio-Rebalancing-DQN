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
        random_ret = np.random.multivariate_normal(self.mu, self.sigma_mat, size=1)
        new_state = new_state * (1+random_ret)
        new_state = new_state / np.sum(new_state) * 100
        new_state = np.round(new_state)
        new_state = (new_state / np.sum(new_state) * 100)
        new_state = np.round(new_state).astype(int)
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

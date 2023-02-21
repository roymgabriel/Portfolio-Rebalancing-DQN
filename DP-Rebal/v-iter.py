import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
import scipy


def setup():
    data_ret = pd.read_csv('Data.csv')

    # Calculate Daily Returns for Assets A & B using Vectoring
    data_ret['Returns_A'] = data_ret['Close_A']/data_ret['Close_A'].shift(1)-1
    data_ret['Returns_B'] = data_ret['Close_B']/data_ret['Close_B'].shift(1)-1
    # Replace NANs with zeros in first line (since there are no returns in the first period)
    data_ret['Returns_A'].fillna(0, inplace=True)
    data_ret['Returns_B'].fillna(0, inplace=True)

    # Define detail of weight increments
    detail = 0.01
    wgt_range = range(0.42, 0.6, detail)

    # Define Amount Invested in Period 0
    initial_amount_invested = 1e6
    # Define Trading Costs of Asset A (in basis points)
    CA = 60
    # Define Trading Costs of Asset B (in basis points)
    CB = 40


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
    def __int__(self, mu, sigma_mat, transaction_cost, gamma):
        self.mu = mu
        self.sigma_mat = sigma_mat
        self.transaction_cost = transaction_cost
        self.gamma = gamma
        x = np.arange(-7, 8)
        self.action_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        self.action_possible = self.action_possible[self.action_possible.sum(axis=1) == 0, :]
        x = np.arange(0, 101)
        self.state_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        self.state_possible = self.state_possible[self.state_possible.sum(axis=1) == 100, :]
        self.value_table = np.zeros(self.state_possible.shape[0])
        self.q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))

    def get_transition_prob(self, state_current, action):
        state_new_wgt = state_current + action
        ret_drift = self.state_possible / state_new_wgt
        # this is only an approximation, it hasn't considered the weight renormalization
        probabilities = scipy.stats.multivariate_normal.pdf(ret_drift, mean=self.mu, cov=self.sigma_mat)
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_value(self, state_wgt):
        action_value_current_state = []
        for action_id in range(self.action_possible.shape[0]):
            action = self.action_possible[action_id]
            transition_prob = self.get_transition_prob(state_wgt, action)
            reward = -expected_cost_total(state_wgt, action, self.mu, self.sigma_mat, self.transaction_cost)
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

        check_converged = np.sub(np.abs(self.value_table - new_value_table))
        self.value_table = new_value_table
        self.q_table = new_q_table

        return check_converged



# Q-values
# Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
# where alpha is the learning rate (a small positive value that determines the weight given to new observations), r is the reward received for taking action a in state s,
# gamma is the discount factor (a value between 0 and 1 that determines the importance of future rewards), s' is the next state, and a' is the optimal action to take in state s' (according to the current Q-values).

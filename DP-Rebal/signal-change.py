import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
import scipy
import random


def reward_sharpe_net_tc(w0, d1, mu, cov, tc):
    w2 = w0 + d1

    # optimal net sharpe
    net_sharpe2 = (w2.dot(mu) - np.sum(np.abs(d1) * tc)) / np.sqrt(w2.dot(cov).dot(w2))

    return net_sharpe2


# two-asset model with dynamic programing and q-table learning

# Bellman equation
# V(s) = max_a sum_s' P(s,a,s') * (r(s,a,s') + gamma * V(s'))
# where P(s,a,s') is the probability of transitioning from state s to state s' with action a, and gamma is a discount factor that determines the importance of future rewards.
# V(s) = max_a E_s'[ r(s,a,s') + gamma * V(s') ]
class BellmanValue:
    # mu will change from period to period through mu_change_cov, which can be set through one-year forward PTVA study
    def __init__(self, mu_init, sigma_mat, mu_change_cov, transaction_cost, gamma):
        self.mu_init = mu_init
        self.sigma_mat = sigma_mat
        self.mu_change_cov = mu_change_cov
        self.transaction_cost = transaction_cost
        self.gamma = gamma
        self.num_asset = len(mu_init)
        x = np.arange(-7, 8)
        self.action_possible = np.array(np.meshgrid(*([x] * len(mu_init)))).T.reshape(-1, len(mu_init))
        self.action_possible = self.action_possible[self.action_possible.sum(axis=1) == 0, :]
        x1 = np.arange(1, 101)
        x2 = np.arange(0, 401, 5)
        self.state_possible = np.array(np.meshgrid(*([x1] * len(mu_init) + [x2] * len(mu_init)))).T.reshape(-1, 2*len(mu_init))
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
        prob_on_wgt = scipy.stats.multivariate_normal.logpdf(ret_drift, mean=state_current[self.state_col_mu]/100, cov=self.sigma_mat)
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
                reward = reward_sharpe_net_tc(state_wgt[self.state_col_wgt]/100, action/100, state_wgt[self.state_col_mu]/100, self.sigma_mat, self.transaction_cost)
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
pvta_sd = np.array([50, 50])
mu_change_cov = np.diag(pvta_sd ** 2)

self = bell = BellmanValue(mu, cov, mu_change_cov, trans_cost, gamma=0.9)
for dummy in range(200):
    diff = bell.iterate_q_table_once()
    if diff < 1e-4:
        break
    print("Iter {}: Value {}".format(dummy, diff))

#######################################################################################################################
# This program will Define a Dynamic Programming algorithm to optimally rebalance a portfolio given the number of assets
# with constant mean and covariance.
#######################################################################################################################


#######################################################################################################################
# Import necessary Libraries
#######################################################################################################################

import scipy.stats as ss
from utils import *

#######################################################################################################################
# Bellman equation
# V(s) = max_a sum_s' P(s,a,s') * (r(s,a,s') + gamma * V(s'))
# where P(s,a,s') is the probability of transitioning from state s to state s' with action a, and gamma is a discount
# factor that determines the importance of future rewards.
# V(s) = max_a E_s'[ r(s,a,s') + gamma * V(s') ]

# it assumes fix mu and sigma_mat
# to consider varying mu and sigma_mat, state space must be expanded, and expectation must consider future new mu
# and sigma_mat
#######################################################################################################################


class BellmanValue:
    def __init__(self, mu, sigma_mat, transaction_cost, gamma, scaling_factor, num_iters=10000):
        self.mu = mu
        self.sigma_mat = sigma_mat
        self.transaction_cost = transaction_cost
        self.gamma = gamma
        # x = np.arange(-99, 100)  # discretize action space
        # self.action_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        # self.action_possible = self.action_possible[self.action_possible.sum(axis=1) == 0, :] / 100
        self.action_possible = np.array([0, 1])
        x = np.arange(1, 101)  # discretize state space
        self.state_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        self.state_possible = self.state_possible[self.state_possible.sum(axis=1) == 100, :] / 100
        self.value_table = np.zeros(self.state_possible.shape[0])
        self.q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))
        self.optimal_weight = find_optimal_wgt(mu, sigma_mat).round(2)
        self.scaling_factor = scaling_factor
        self.num_iters = num_iters
        # self.tc_cum_cost = {key: [] for key in self.action_possible}
        self.tc_cum_cost = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))

    def get_transition_prob(self, state_wgt):
        # ret_drift = self.state_possible / state_wgt  # [20, 80] / [40, 60]
        # (w_t - w_t+1) / w_t+1
        # [50, 50] -> [60, 40] -> [100, 0]
        # ret_drift -= 1

        # Pratyush suggestion
        # return drift should be ((current_weight + action)/current_weight) - 1
        # so for binary case we should have (current_weight + action) to be optimal_weight
        # and our current weight is state_wgt
        # also if we do not rebalance then return drift is just 0
        # or more precisely (current_weight+action) = current_weight
        ret_drift = (self.optimal_weight/state_wgt) - 1
        ret_drift = np.vstack([ret_drift, np.zeros(len(self.mu))])

        # this is only an approximation, it hasn't considered the weight renormalization
        # mean=self.optimal_weight / state_wgt - 1
        probabilities = ss.multivariate_normal.pdf(ret_drift, mean=self.mu, cov=self.sigma_mat)
        probabilities /= np.sum(probabilities)
        # probabilities = np.zeros(len(self.state_possible))
        # idx = np.argmin(np.abs(self.state_possible[:, 0] - state_wgt[0]))
        # probabilities[idx] = 1
        return probabilities

    def calculate_value(self, state_wgt, state_id):
        action_value_current_state = []
        for action_id in range(self.action_possible.shape[0]):
            action = self.action_possible[action_id]
            if action == 0:
                new_state = state_wgt
            elif action == 1:
                new_state = self.optimal_weight
            else:
                raise ValueError(f"Action value of {action} is invalid!")

            if np.any(new_state <= 0):
                # action_value = -np.inf
                raise ValueError(f"New state of {new_state} is invalid!")
            else:
                transition_prob = self.get_transition_prob(new_state)
                # reward = -expected_cost_total(w0=state_wgt, w1=new_state, opt_w=self.optimal_weight, mu=self.mu,
                #                               cov=self.sigma_mat, tc=self.transaction_cost)
                # reward = -obj_func(x=new_state, mu=self.mu, cov=self.sigma_mat)
                cost_i = cost_turnover(w0=state_wgt, w1=new_state, tc=self.transaction_cost) * self.scaling_factor
                # self.tc_cum_cost[state_id, action] += cost_i
                sharpe = net_sharpe(w1=new_state, mu=self.mu, cov=self.sigma_mat)
                reward = sharpe - cost_i  #  np.sum(self.tc_cum_cost[state_id, action])
                print(f"reward = {reward} | action = {action} | tc cost = {cost_i}")
                next_state_value = self.value_table[state_id]
                action_value = np.sum(transition_prob * (reward + self.gamma * next_state_value))
            action_value_current_state.append(action_value)
        return action_value_current_state

    def iterate_q_table_once(self):
        new_value_table = np.zeros(self.state_possible.shape[0])
        new_q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))
        for state_id in range(self.state_possible.shape[0]):
            state_wgt = self.state_possible[state_id]
            new_q_table[state_id, :] = self.calculate_value(state_wgt, state_id)
            new_value_table[state_id] = np.max(new_q_table[state_id, :])

        check_converged = np.sum(np.abs(self.value_table - new_value_table))
        self.value_table = new_value_table
        self.q_table = new_q_table

        return check_converged

    def iterate(self):
        print("Iteration Start:\n")
        for d in range(self.num_iters):
            diff = self.iterate_q_table_once()
            if diff < 1e-9:
                print(f"n_asset = {len(self.mu)}  | tc = {self.transaction_cost} | Iteration finished at step {d}.")
                break
            print(f"\t iter {d}: n_assets = {len(self.mu)}  |  tc = {self.transaction_cost}  | Value {diff}")

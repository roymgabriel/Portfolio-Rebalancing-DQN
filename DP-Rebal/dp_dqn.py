import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as mpl
import scipy.stats as ss
from scipy.optimize import minimize

mpl.rcParams.update({"font.size": 18})


def net_sharpe(w1, mu, cov, w0, tc):
    return (w1.dot(mu) - cost_turnover(w0, w1, tc)) / np.sqrt(w1.dot(cov).dot(w1))


def obj_func(x, mu, cov):
    return -x.dot(mu) / np.sqrt(x.dot(cov).dot(x))
    # return 0.5 * (x.dot(cov).dot(x)) - x.dot(mu)


def find_optimal_wgt(mu, cov):
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
    return np.sum(np.abs(w1 - w0) * tc) / 2


def expected_cost_total(w0, w1, opt_w, mu, cov, tc):
    opt_net_sharpe = net_sharpe(opt_w, mu, cov, w0, tc)
    w1_net_sharpe = net_sharpe(w1, mu, cov, w0, tc)
    return opt_net_sharpe - w1_net_sharpe


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
        x = np.arange(-99, 100)
        self.action_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        self.action_possible = self.action_possible[self.action_possible.sum(axis=1) == 0, :] / 100
        x = np.arange(1, 101)
        self.state_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu))
        self.state_possible = self.state_possible[self.state_possible.sum(axis=1) == 100, :] / 100
        self.value_table = np.zeros(self.state_possible.shape[0])
        self.q_table = np.zeros((self.state_possible.shape[0], self.action_possible.shape[0]))
        self.optimal_weight = find_optimal_wgt(mu, sigma_mat)

    def get_transition_prob(self, state_wgt):
        ret_drift = self.state_possible / state_wgt
        ret_drift -= 1
        # this is only an approximation, it hasn't considered the weight renormalization
        probabilities = ss.multivariate_normal.pdf(ret_drift, mean=self.mu, cov=self.sigma_mat)
        probabilities /= np.sum(probabilities)
        # probabilities = np.zeros(len(self.state_possible))
        # idx = np.argmin(np.abs(self.state_possible[:, 0] - state_wgt[0]))
        # probabilities[idx] = 1
        return probabilities

    def calculate_value(self, state_wgt):
        action_value_current_state = []
        for action_id in range(self.action_possible.shape[0]):
            action = self.action_possible[action_id]
            new_state = state_wgt + action
            if np.any(new_state <= 0):
                action_value = -np.inf
            else:
                transition_prob = self.get_transition_prob(new_state)
                reward = -expected_cost_total(state_wgt, new_state, self.optimal_weight, self.mu, self.sigma_mat,
                                              self.transaction_cost)
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

    def iterate(self):
        print("Iteration Start")
        for dummy in range(500):
            diff = self.iterate_q_table_once()
            if diff < 1e-5:
                print(f"Iteration finish at step {dummy}")
                break
            print("\t iter {}: Value {}".format(dummy, diff))


n_manager = 2
mu = np.array([50, 200]) / 1e4
sigma = np.array([300, 800]) / 1e4
cov = np.diag(sigma ** 2)
optimal_weight = find_optimal_wgt(mu, cov)
x0 = np.ones(len(mu)) / len(mu)

model_result = {}
for tc in [0, 0.0005, 0.001, 0.002]:
    self = bell = BellmanValue(mu, cov, tc, gamma=0.9)
    bell.iterate()
    model_result[tc] = bell

x = self.state_possible[:, 0]
action_df = pd.DataFrame(index=x)
action_bm_df = pd.DataFrame(index=x)
for tc, bell in model_result.items():
    # visualize q table
    action = np.array([bell.action_possible[i, 0] for i in bell.q_table.argmax(axis=1)])
    action_df[f"TC: {tc * 1e4:.0f} bps"] = action
    action_bm = []
    for i in x:
        a = x[np.argmax([net_sharpe(np.array([j, 1 - j]), mu, cov, np.array([i, 1 - i]), tc) for j in x])] - i
        action_bm.append(a)
    action_bm = np.array(action_bm)
    action_bm_df[f"TC: {tc * 1e4:.0f} bps BM"] = action_bm

fig, ax = mpl.subplots(1, 1, figsize=(20, 10))
action_df.plot(ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
action_bm_df.plot(ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], linestyle="dashed")
ax.set_xlabel("Weight on First Asset")
ax.set_ylabel("Suggested Delta Weight on Asset 1")
ax.axvline(optimal_weight[0], color="red", linestyle="dotted")
ax.axhline(0, color="red", linestyle="dotted")
ax.legend()
mpl.tight_layout()
mpl.show()
mpl.savefig(os.path.expanduser(f"~/Desktop/q_table.png"))
mpl.close()

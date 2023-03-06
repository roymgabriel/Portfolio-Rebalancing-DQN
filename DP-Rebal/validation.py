import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as mpl
import scipy.stats as ss
from scipy.optimize import minimize
from signal_change import DQNlearning
import random

mpl.rcParams.update({"font.size": 18})


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


def cost_turnover(tc, w1, w0):
    return np.sum(np.abs(w1 - w0) * tc) / 2


def cost_suboptimality(w0, mu, cov):
    w1 = find_optimal_wgt(mu, cov)
    ret_ce_optimal = obj_func(w1, mu, cov)
    ret_ce_current = obj_func(w0, mu, cov)
    return ret_ce_optimal - ret_ce_current


# def cost_net_sharpe(optimal_weight, w0, d1, mu, cov, tc):
#     w1 = optimal_weight
#     w2 = w0 + d1
#
#     # optimal net sharpe
#     net_sharpe1 = (w1.dot(mu) - cost_turnover(tc, w1, w0)) / np.sqrt(w1.dot(cov).dot(w1))
#     net_sharpe2 = (w2.dot(mu) - cost_turnover(tc, w2, w0)) / np.sqrt(w2.dot(cov).dot(w2))
#
#     return net_sharpe1 - net_sharpe2

def cost_net_sharpe(w0, d1, mu, cov, tc):
    w1 = optimal_weight # global optimal weight to speed up computation
    w2 = w0 + d1

    # optimal net sharpe
    net_sharpe1 = (w1.dot(mu) - cost_turnover(tc, w1, w0)) / np.sqrt(w1.dot(cov).dot(w1))
    net_sharpe2 = (w2.dot(mu) - cost_turnover(tc, w2, w0)) / np.sqrt(w2.dot(cov).dot(w2))

    return net_sharpe1 - net_sharpe2

def expected_cost_total(state_wgt, action_delta, mu, sigma_mat, transaction_cost):
    # cost_p1 = cost_turnover(transaction_cost, state_wgt, state_wgt + action_delta)
    # cost_p2 = cost_suboptimality(state_wgt + action_delta, mu, sigma_mat)
    # return cost_p1 + cost_p2

    # change to net sharpe
    cost = cost_net_sharpe(state_wgt, action_delta, mu, sigma_mat, transaction_cost)
    return cost


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
        probabilities = ss.multivariate_normal.pdf(ret_drift, mean=self.mu, cov=self.sigma_mat)
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
                reward = -expected_cost_total(state_wgt / 100, action / 100, self.mu, self.sigma_mat,
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


# 2 manager examples
n_manager = 2
mu = np.array([50, 200]) / 1e4
sigma = np.array([300, 800]) / 1e4
cov = np.diag(sigma ** 2)
tc = 0.001
optimal_weight = find_optimal_wgt(mu, cov)
x0 = np.ones(len(mu)) / len(mu)
self = bell = BellmanValue(mu, cov, tc, gamma=0.9)
for dummy in range(200):
    diff = bell.iterate_q_table_once()
    if diff < 1e-5:
        break
    print("Iter {}: Value {}".format(dummy, diff))

y = [-cost_net_sharpe(np.array([i / 100, 1 - i / 100]), np.array([0, 0]), mu, cov, tc) for i in range(99)]
mpl.figure(1, figsize=(20, 10))
mpl.plot(np.arange(99) / 100, self.value_table, label="Value Table")
mpl.plot(np.arange(99) / 100, y, label="Net Sharpe Loss")
mpl.xlabel("Weight on First Asset")
mpl.axvline(optimal_weight[0], label="Optimal", color="red")
mpl.legend()
mpl.tight_layout()
mpl.savefig('fig1.png')
mpl.show()

# setup and training for DQN
num_asset = 2
pvta_sd = np.array([50, 50])
mu_change_cov = np.diag(pvta_sd ** 2)

dqn = DQNlearning(num_asset, cov, mu_change_cov, tc, gamma=0.9, epsilon=0.1, learning_rate=0.001)

num_episodes = 1000
max_steps_per_episode = 100

for i in range(num_episodes):
    print("Epoch {}".format(i))
    current_state = random.randint(0, dqn.num_states - 1)
    for j in range(max_steps_per_episode):
        current_state = dqn.network_training_once(current_state)

import pickle
with open('dqn.pkl', 'wb') as f:
    pickle.dump(dqn, f)

y = [-cost_net_sharpe(np.array([i / 100, 1 - i / 100]), np.array([0, 0]), mu, cov, tc) for i in range(99)]
mpl.clf()
mpl.figure(1, figsize=(20, 10))
mpl.plot(np.arange(99) / 100, bell.value_table - np.max(bell.value_table), label="Value Table")
mpl.plot(np.arange(99) / 100, y - np.max(y), label="Net Sharpe")
mpl.xlabel("Weight on First Asset")
mpl.axvline(optimal_weight[0], label="Optimal", color="red")


chunk_size = 99
i = 900
y = dqn.value_table[(i-1)*chunk_size:i*chunk_size]
mpl.plot(np.arange(99) / 100, y - np.max(y), label="mu (40, 50)")
print(dqn.state_possible[(i-1)*chunk_size:i*chunk_size, :])

mpl.legend()
mpl.tight_layout()
mpl.savefig('figdqn2.png')
mpl.show()



# # simulation for more managers
# n_manager = 15
# mu = np.random.uniform(75, 120, n_manager) / 1e4
# ir = np.random.uniform(0.3, 0.4, n_manager)
# sigma = mu / ir
# cov = np.diag(sigma ** 2)
# optimal_weight = find_optimal_wgt(mu, cov)
# tc = 0.001
n_hist = 24
n_sample = 1000

start = dt.datetime(2013, 1, 1)
end = dt.datetime(2022, 12, 31)
dates = pd.date_range(start, end, freq="M")
test_dates = dates[n_hist:]

# use real manager data
# ptva_df = pd.read_parquet(os.path.expanduser("~/Desktop/us_ev_data/na_ptva.pgz"))
# rtva_df = pd.read_parquet(os.path.expanduser("~/Desktop/us_ev_data/na_rtva.pgz"))
# alpha_df = pd.read_parquet(os.path.expanduser("~/Desktop/us_ev_data/na_alpha.pgz")).loc[start:end]
# exclude short history and all 0 managers and extreme or high ptva managers
# valid_mask = ptva_df.notnull().sum().sort_values() >= 24
# non_negative_mask = (ptva_df.mean() > 10)
# non_extreme_mask = (ptva_df.mean() < 500)
# ptva_df = ptva_df.loc[:, valid_mask & non_negative_mask & non_extreme_mask]
# rtva_df = rtva_df.loc[:, valid_mask & non_negative_mask & non_extreme_mask]

# sample_evids = ptva_df.mean()[ptva_df.mean() >= 0].index
# sample_evids = rtva_df.columns[rtva_df.notnull().all()].intersection(alpha_df.columns)

# check correlation
# hist_ret = rtva_df[sample_evids].expanding(24).mean() * 12e4
# two_year_ret = rtva_df[sample_evids].rolling(24).mean() * 12e4
# fut_ret = rtva_df[sample_evids].rolling(6).mean().shift(-6) * 12e4
# compare_df = pd.merge(
#     hist_ret.reset_index().melt(id_vars="index", value_name="hist_ret", var_name="ev_id"),
#     fut_ret.reset_index().melt(id_vars="index", value_name="fut_ret", var_name="ev_id"),
#     on=["index", "ev_id"]
# )
# compare_df = pd.merge(
#     compare_df,
#     two_year_ret.reset_index().melt(id_vars="index", value_name="2y_ret", var_name="ev_id"),
#     on=["index", "ev_id"]
# )
# compare_df = pd.merge(
#     compare_df,
#     ptva_df.reset_index().melt(id_vars="index", value_name="ptva", var_name="ev_id"),
#     on=["index", "ev_id"]
# )
# compare_df = compare_df.dropna()
# compare_df["hist_ret_non_negative"] = np.maximum(0, compare_df["hist_ret"])
#
# correlation_df = compare_df.groupby("index")[["hist_ret", "2y_ret", "fut_ret", "ptva", "hist_ret_non_negative"]].apply(
#     lambda x: x.corr().loc["fut_ret"]).drop(columns="fut_ret")
# print(correlation_df.mean())

#
# fig, axes = mpl.subplots(2, 1, figsize=(20, 10))
# ax = axes[0]
# correlation_df.plot(ax=ax)
# ax.set_title("Monthly CXC with Future 6m")
# ax = axes[1]
# correlation_df.rolling(6).mean().plot(ax=ax)
# ax.set_title("Rolling 6M CXC with Future 6m")
# mpl.tight_layout()
# mpl.savefig(os.path.expanduser("~/Desktop/tva_correlation.png"))
# mpl.close()

ret_result = []
net_ret_result = []
sharpe_result = []
net_sharpe_result = []
turnover_result = []

# mean 0, delta sd 30, error 50 - 100
mu_init = mu
ptva_delta_mean = 0
ptva_delta_sd = 30 / 1e4
ptva_error = 70 / 1e4


for _k in range(n_sample):
    if _k % 100 == 99:
        print(f"Running {_k + 1}/{n_sample}")

    # simulate mu and sigma from real managers
    # n_manager = 15
    # evids = np.random.choice(sample_evids, n_manager)
    # mu = ptva_df.mean().loc[evids].values / 1e4
    # sigma = rtva_df.std().loc[evids].values * np.sqrt(12)
    # cov = np.diag(sigma ** 2)

    # simulate return
    mu = []
    mu_curr = mu_init
    mu.append(mu_curr)
    for t in range(1, len(dates)):
        mu_curr = np.random.normal(mu_curr + ptva_delta_mean, ptva_delta_sd, size=len(mu_curr))
        mu_curr = np.maximum(0, mu_curr)
        mu.append(mu_curr)
    mu = np.array(mu)
    # mpl.plot(mu)
    # mpl.show()

    # ret = np.random.multivariate_normal(mu / 12, cov / 12, size=len(dates))
    ret = []
    for t in range(0, len(dates)):
        ret.append(np.random.multivariate_normal(mu[t, :] / 12, cov / 12))
    ret = np.array(ret)

    ret_df = pd.DataFrame(ret, index=dates)
    # ret_df = rtva_df[evids]
    # mu_df = ptva_df[evids].reindex(ret_df.index)
    # ret_df.add(1).cumprod().plot(figsize=(20, 10))
    # mpl.show()

    # use past to estimate
    mu_est = np.maximum(0, ret_df.iloc[:n_hist].mean().values * 12)
    # mu_est = mu_df.iloc[n_hist-1].values / 1e4
    sigma_est = ret_df.iloc[:n_hist].std().values * np.sqrt(12)
    cov_est = np.diag(sigma_est ** 2)
    opt_weight = find_optimal_wgt(mu_est, cov_est)
    opt_weight = optimal_weight

    weight = np.zeros((len(test_dates), n_manager))
    weight[0, :] = opt_weight
    market = ret_df.iloc[n_hist:].values

    wgt_dict = {
        "No Rebalance": weight.copy(),
        "Monthly Rebalance": weight.copy(),
        "Quarterly Rebalance": weight.copy(),
        "Yearly Rebalance": weight.copy(),
        "3% Rebalance": weight.copy(),
        "5% Rebalance": weight.copy(),
        "8% Rebalance": weight.copy(),
        "DP Rebalance": weight.copy(),
        "DQN Rebalance": weight.copy(),
    }
    to_dict = {k: np.zeros(len(test_dates)) for k in wgt_dict.keys()}
    for t in range(1, weight.shape[0]):
        # update optimal weight by using expanding window to reestimate mu and sigma
        mu_est = np.maximum(0, ret_df.iloc[(t + n_hist - 24):(t + n_hist)].mean().values * 12)
        # mu_est = mu_df.iloc[t + n_hist - 1].values / 1e4
        sigma_est = ret_df.iloc[(t + n_hist - 24):(t + n_hist)].std().values * np.sqrt(12)
        cov_est = np.diag(sigma_est ** 2)
        opt_weight = find_optimal_wgt(mu_est, cov_est)
        # opt_weight = optimal_weight

        r = market[t - 1]
        for k, wgt in wgt_dict.items():
            w = wgt[t - 1]
            # drift by market
            w0 = w * (1 + r) / (w * (1 + r)).sum()
            if k == "No Rebalance":
                w1 = w0
            elif k == "Monthly Rebalance":
                w1 = opt_weight
            elif k == "Quarterly Rebalance":
                w1 = opt_weight if t % 3 == 0 else w0
            elif k == "Yearly Rebalance":
                w1 = opt_weight if t % 12 == 0 else w0
            elif k == "3% Rebalance":
                w1 = opt_weight if np.sum(np.abs(w0 - opt_weight)) / 2 > 0.03 else w0
            elif k == "5% Rebalance":
                w1 = opt_weight if np.sum(np.abs(w0 - opt_weight)) / 2 > 0.05 else w0
            elif k == "8% Rebalance":
                w1 = opt_weight if np.sum(np.abs(w0 - opt_weight)) / 2 > 0.08 else w0
            elif k == "DP Rebalance":  # has true mu and cov information
                state_id = int(round(w0[0] * 100, 0)) - 1
                action_id = np.where(bell.q_table[state_id] == bell.q_table[state_id].max())[0][0]
                w1 = (bell.state_possible[state_id] + bell.action_possible[action_id]) / 100
                w1 = w0 if np.max(w1 - w0) < 0.005 else w1
            elif k == "DQN Rebalance":  # has true cov information
                action = dqn.find_best_action(np.concatenate([w0.reshape([-1]) * 100, mu_est.reshape([-1]) * 1e4]))
                action_wgt = action['wgt_delta']
                w1 = (round(w0[0] * 100) + action_wgt) / 100
                w1 = w0 if np.max(w1 - w0) < 0.005 else w1

            wgt[t] = w1
            to_dict[k][t] = np.sum(np.abs(w1 - w0)) / 2

    ret_compare_df = pd.concat([
        pd.Series((wgt_dict[k] * market).sum(axis=1), index=test_dates, name=k) for k in wgt_dict.keys()
    ], axis=1)
    turnover_df = pd.concat([
        pd.Series(to_dict[k], index=test_dates, name=k) for k in to_dict.keys()
    ], axis=1)
    net_ret_compare_df = ret_compare_df - turnover_df * tc

    ret_result.append(ret_compare_df.mean() * 12)
    net_ret_result.append(net_ret_compare_df.mean() * 12)
    sharpe_result.append(ret_compare_df.mean() / ret_compare_df.std() * np.sqrt(12))
    net_sharpe_result.append(net_ret_compare_df.mean() / net_ret_compare_df.std() * np.sqrt(12))
    turnover_result.append(turnover_df.mean() * 12)

turnover_summary_df = pd.concat(turnover_result, axis=1).transpose()
ret_summary_df = pd.concat(ret_result, axis=1).transpose()
net_ret_summary_df = pd.concat(net_ret_result, axis=1).transpose()
sharpe_summary_df = pd.concat(sharpe_result, axis=1).transpose()
net_sharpe_summary_df = pd.concat(net_sharpe_result, axis=1).transpose()

fig, axes = mpl.subplots(3, 2, figsize=(40, 20))
ax = axes[0, 0]
ret_summary_df.plot.hist(ax=ax, alpha=0.2, bins=20)
ax.set_title("Annualized Performance")
ax.legend([f"{c}: {ret_summary_df[c].mean() * 1e4:.1f} bps" for c in ret_summary_df.columns])

ax = axes[0, 1]
ret_summary_df.plot.hist(ax=ax, alpha=0.2, bins=20)
ax.set_title("Annualized Performance (Net)")
ax.legend([f"{c}: {net_ret_summary_df[c].mean() * 1e4:.1f} bps" for c in net_ret_summary_df.columns])

ax = axes[1, 0]
ret_summary_df.plot.hist(ax=ax, alpha=0.2, bins=20)
ax.set_title("Annualized Sharpe")
ax.legend([f"{c}: {sharpe_summary_df[c].mean():.3f}" for c in sharpe_summary_df.columns])

ax = axes[1, 1]
ret_summary_df.plot.hist(ax=ax, alpha=0.2, bins=20)
ax.set_title("Annualized Sharpe (Net)")
ax.legend([f"{c}: {net_sharpe_summary_df[c].mean():.3f}" for c in net_sharpe_summary_df.columns])

ax = axes[2, 0]
turnover_summary_df.plot.hist(ax=ax, alpha=0.2, bins=20)
ax.set_title("Annualized Turnover")
ax.legend([f"{c}: {turnover_summary_df[c].mean():.1%}" for c in turnover_summary_df.columns])

ax = axes[2, 1]
turnover_summary_df.plot.hist(ax=ax, alpha=0.2, bins=20)
ax.set_title("Annualized Transaction Cost")
ax.legend([f"{c}: {turnover_summary_df[c].mean() * tc * 1e4:.2f} bps" for c in turnover_summary_df.columns])

mpl.tight_layout()
mpl.show()
mpl.savefig('comparison_result_with_sig_change.png')

stats = pd.concat([
    (ret_summary_df.mean() * 1e4).round(1), sharpe_summary_df.mean().round(3),
    (turnover_summary_df.mean() * 100).round(1), (net_ret_summary_df.mean() * 1e4).round(1),
    net_sharpe_summary_df.mean().round(3)], axis=1
).to_csv(os.path.expanduser("~/Desktop/stats.csv"))

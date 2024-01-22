import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as ss
import time

import warnings

warnings.filterwarnings('ignore')


class Bellman2():
    def __init__(self, mu, cov, transaction_cost, gamma, wn=100, precision=2):
        self.wn = wn
        self.mu = mu
        self.n = len(mu)
        self.cov = cov
        self.transaction_cost = transaction_cost
        self.gamma = gamma
        self.precision = precision
        self.optimal_weight = np.round(self.get_optimal_weights(mu, cov),self.precision)
        print(f"optimal weight: {self.optimal_weight}")
        self.states = np.round(self.get_state_space(), self.precision)
        print(f"states: {self.states.shape}")
        self.states_indices = self.states.shape[0]
        self.value_table = np.zeros(self.states.shape[0])


    def get_state_space(self):
        '''
        w_0: array, initial portfolio weights
        ---
        returns: array, state space
        '''
        w = np.asarray(np.meshgrid(*[np.linspace(0, 1, self.wn + 1) for _ in range(self.n)])).T.reshape(-1, self.n)
        w = w[np.abs(np.sum(w, axis=1) - 1) < 1e-6]
        return w

    def get_action_space(self):
        '''
        n: int, number of assets
        ---
        returns: array, action space
        '''
        a = np.asarray(np.meshgrid(*[np.linspace(-1, 1, 2 * self.wn + 1) for _ in range(self.n)])).T.reshape(-1, self.n)
        a = a[np.abs(np.sum(a, axis=1)) < 1e-6]
        return a

    def get_transaction_cost(self, c, w, w_next):
        '''
        c: array, transaction cost per dollar
        w: array, current portfolio weights
        w_next: array, next period portfolio weights
        ---
        returns: float, transaction cost
        '''
        return np.sum(c * np.abs(w_next - w))

    def get_optimal_weights(self, mu, cov):
        '''
        w: array, current portfolio weights
        mu: array, expected returns
        cov: array, covariance matrix
        ---
        returns: array, optimal portfolio weights
        '''

        # Define the objective function to minimize
        def objective(w, mu, cov):
            return -w.dot(mu) / np.sqrt(w.dot(cov).dot(w))

        # Define the equality constraint: sum_i w_i = 1
        def constraint_equal(w):
            return 1 - np.sum(w)

        # Define the bounds for w: w >= 0
        bounds = [[0, 1] for _ in range(len(mu))]

        # Initial guess for w
        initial_w = np.array(np.ones(len(mu)) / len(mu))

        # Define the optimization problem
        opt_result = minimize(objective, initial_w, args=(mu, cov), constraints=(
            {'type': 'eq', 'fun': constraint_equal}),
                              tol=1e-6,
                              bounds=bounds,
                              options={"maxiter": 10000}
                              )
        return opt_result.x

    def cost_net_sharpe(self, current_state):
        w_opt = self.optimal_weight  # global optimal weight to speed up computation
        next_states = np.round(self.states, self.precision)
        net_sharpe1 = (w_opt.dot(self.mu) - self.get_transaction_cost(self.transaction_cost, w_opt, current_state)) / np.sqrt(w_opt.dot(self.cov).dot(w_opt))
        net_sharpe2 = ((next_states).dot(self.mu) - np.sum(self.transaction_cost * np.abs(self.states-current_state), axis=1)) / np.sqrt(np.sum(np.multiply(np.matmul((next_states),self.cov), (next_states)),axis=1))
        return net_sharpe1 - net_sharpe2

    def cost_total(self, current_state):
        return self.cost_net_sharpe(current_state)

    def get_transition_prob(self, state_current):
        state_current_temp = np.where(np.abs(state_current) <1e-4, 0.005, state_current)
        ret_drift = self.states/state_current_temp - 1 # self.actions/state_current_temp
        # ret_drift = self.states/state_current - 1
        probabilities = ss.multivariate_normal.pdf(ret_drift, mean=self.mu, cov=self.cov)
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_value(self, current_state):
        transition_prob = self.get_transition_prob(current_state)
        reward = -self.cost_total(current_state)
        next_state_value = self.value_table
        action_value_temp=np.sum(transition_prob.reshape(-1,1)*reward.reshape(1,-1) + self.gamma*(transition_prob*next_state_value).reshape(-1,1), axis=0)
        action = np.round(self.states[np.argmax(action_value_temp)]-current_state, self.precision)
        return (np.max(action_value_temp), action)

    def iterate_q_table_once(self):
        new_value_table = np.zeros(self.states.shape[0])
        self.optimum_actions = np.zeros((self.states.shape[0], self.n))


        for state_id in range(self.states.shape[0]):
            # print(1)
            # if state_id % 100 == 0:
                # print(f"State progress: {state_id}/{self.states.shape[0]}")
            state_wgt = self.states[state_id]
            action_value, optimum_action = self.calculate_value(state_wgt)
            # print(2)
            new_value_table[state_id] = action_value
            self.optimum_actions[state_id] = optimum_action

        check_converged = np.sum(np.abs(self.value_table - new_value_table))
        self.value_table = new_value_table

        return check_converged


# n_asset_lt= [2,4]#,8,16]
ITERATIONS=1
n_asset_lt = [3]  # ,8,16]
asset_dic = dict()
for n_asset in n_asset_lt:
    start_time = time.time()
    mu = np.linspace(50, 200, n_asset) / 1e4
    sigma = np.linspace(300, 800, n_asset) / 1e4
    cov = np.diag(sigma ** 2)
    transaction_cost_lt = [0, 0.0005, 0.001]
    opts = []
    for tc in transaction_cost_lt:
        bell = Bellman2(mu, cov, tc, gamma=0.9, wn=100, precision=2)
        for dummy in range(ITERATIONS):
            diff = bell.iterate_q_table_once()
            if diff < 1e-3:
                break
            if dummy % 100 == 0:
                print(f"\t iter {dummy}: n_assets = {n_asset}  |  tc = {tc}  | Value {diff}")
        # create a df of bell.optimum_actions and bell.states and save it to csv
        df = pd.DataFrame(bell.optimum_actions)
        df['state'] = bell.states.tolist()
        df.to_csv(f'output/assets_{n_asset}_tc_{tc}_iter_{ITERATIONS}.csv')


        opts.append(bell.optimum_actions[:, 0])
        time_spent = time.time() - start_time
        print(f"asset: {n_asset}, tc:{tc}, time spent:{time_spent} seconds")
        plt.title(f"Assets: {n_asset}")
        for i in range(len(opts)):
            plt.plot(bell.states[:, 0], opts[i], label=f'tc={transaction_cost_lt[i]}')
        plt.legend()
        plt.savefig(f'pics/{n_asset}_assets_{ITERATIONS}_iters.png', dpi=300, bbox_inches='tight')
        del bell
        plt.clf()
    # plt.title(f"Assets: {n_asset}")
    # for i in range(len(transaction_cost_lt)):
    #     plt.plot(bell.states[:, 0], opts[i], label=f'tc={transaction_cost_lt[i]}')
    # plt.legend()
    # plt.savefig(f'pics/{n_asset}_assets.png', dpi=300, bbox_inches='tight')
    # plt.clf()
    # del bell



# path='D:\\Apps\\PycharmProjects\\output\\3_assets_tc_0.0005.csv'
# df=pd.read_csv(path)
# df[['s0','s1','s2']]=df['state'].str.split(",", expand=True)
# df['s0']=df['s0'].str.replace('[','')
# df['s2']=df['s2'].str.replace(']','')
#
#
# import plotly.graph_objects as go
#
# # Create a 3D line plot using Plotly
# fig = go.Figure()
#
# # Add a 3D scatter plot
# fig.add_trace(go.Scatter3d(
#     x=df['s0'],
#     y=df['s1'],
#     z=df['0'],
#     mode='lines+markers',
#     marker=dict(size=2, color='blue'),
#     line=dict(color='darkblue', width=2)
# ))
#
# # Set layout properties
# fig.update_layout(scene=dict(
#                     xaxis_title='s0',
#                     yaxis_title='s1',
#                     zaxis_title='0'),
#                     margin=dict(l=0, r=0, b=0, t=0))
#
# # Show the plot
# fig.show()
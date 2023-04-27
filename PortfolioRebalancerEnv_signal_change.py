import math
from typing import Optional, Union

import numpy as np

import gym
from gym.utils import seeding
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

from scipy.optimize import minimize


def net_sharpe(w1, mu, cov, w0, tc):
    # ROY NOTE: I changed w1 to w1.T in denominator
    return (w1.dot(mu) - cost_turnover(w0, w1, tc)) / np.sqrt(w1.dot(cov).dot(w1.T))


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



class PortfolioRebalancerEnv(gym.Env):
    """
    Description:


    """
    def __init__(self,
                 w_optimal: Union[np.ndarray, list],
                 transaction_costs: Union[np.ndarray, list, float, int] = 0,
                 n_assets: Union[np.float32, np.ndarray, list] = 2,
                 mu_error: Union[np.float32, np.ndarray, list, int] = 0,
                 cov: Union[np.float32, np.ndarray, list] = np.array([300, 800]) / 1e4,
                 mu: Union[np.float32, np.ndarray, list] = np.array([300, 800]) / 1e4,
                 mu_change_cov: Union[np.float32, np.ndarray, list] = np.array([300, 800]) / 1e4,
                 w_min: float = 0.0,
                 w_max: float = 100.0
                 ):

        # Define mean array of returns (or constant)
        self.mu = mu
        self.mu_error = mu_error

        # define number of assets
        self.n_assets = n_assets
        self.input_size = self.n_assets * 2

        # define action parameters
        # x = np.arange(-99, 100)
        x = np.arange(-7, 8)
        self.action_space = np.array(np.meshgrid(*([x] * self.n_assets))).T.reshape(-1, self.n_assets)
        self.action_space = self.action_space[self.action_space.sum(axis=1) == 0, :]



        # define state space (observation space)
        x1 = np.arange(1, 101)
        x2 = np.arange(0, 401, 5)
        self.state_possible = np.array(np.meshgrid(*([x1] * self.n_assets + [x2] * self.n_assets))).T.reshape(-1, 2*self.n_assets)
        self.state_possible = self.state_possible[self.state_possible[:, 0:self.n_assets].sum(axis=1) == 100, :]

        self.action_feasible = list()
        self.state_col_wgt = range(0, self.n_assets)
        self.state_col_mu = range(self.n_assets, 2*self.n_assets)
        for state_id in range(self.n_assets):
            state = self.state_possible[state_id]
            self.action_feasible.append(
                np.argwhere(np.all(state[self.state_col_wgt] + self.action_space > 0, axis=1)).reshape([-1])
            )

        # define boundary conditions
        self.w_max = w_max
        self.w_min = w_min

        self.state = None
        self.steps_beyond_terminated = None
        self.seed()

        # define optimal variance constant
        self.cov = cov
        self.sigma = np.diag(cov)
        self.sigma_mat = np.diag(self.sigma ** 2)

        self.mu_change_cov = mu_change_cov

        # define optimal weight from min variance portfolio
        self.w_optimal = w_optimal

        # define transaction costs array
        self.transaction_costs = transaction_costs



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_next_state(self, action_delta):
        new_state_wgt = self.state.squeeze()[self.state_col_wgt] + action_delta
        mu = self.state.squeeze()[self.state_col_mu] / 1e4 # only change mu's not weights
        random_ret = np.random.multivariate_normal(mu, self.cov, size=1)
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

    def step(self, action_delta):
        err_msg = f"{action_delta!r} ({type(action_delta)}) invalid"
        assert action_delta in self.action_space, err_msg
        assert self.state is not None, "Call reset before using step method."

        new_state = self.get_next_state(action_delta=action_delta)

        terminated = True if np.any(np.any(self.state <= self.w_min) or
                                 np.any(self.state >= self.w_max) or
                                 np.any(new_state <= self.w_min) or
                                 np.any(new_state >= self.w_max)) else False


        if not terminated:
            # calculate reward
            reward = -expected_cost_total(w0=self.state.squeeze()[self.state_col_wgt]/100,
                                          w1=new_state[self.state_col_wgt]/100,
                                          opt_w=self.w_optimal,
                                          mu=self.state.squeeze()[self.state_col_mu]/1e4,
                                          cov=self.cov,
                                          tc=self.transaction_costs)
        elif self.steps_beyond_terminated is None:
            # weights exceeded 0/100 % threshold
            self.steps_beyond_terminated = 0

            # Calculate Total Costs to be incurred if rebalancing is to take place
            reward = -expected_cost_total(w0=self.state.squeeze()[self.state_col_wgt]/100,
                                          w1=new_state[self.state_col_wgt]/100,
                                          opt_w=self.w_optimal,
                                          mu=self.state.squeeze()[self.state_col_mu]/1e4,
                                          cov=self.cov,
                                          tc=self.transaction_costs)

        # if terminated
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

            reward = -1

        return np.array(self.state, dtype=np.float32), reward, terminated

    def reset(self):

        # exclude one asset TODO: needs to be fixed for multi
        row_id = np.random.choice(self.state_possible.shape[0], size=1)
        self.state = self.state_possible[row_id]
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}




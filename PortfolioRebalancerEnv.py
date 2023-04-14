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
    # TODO: Note that the states only have weights of asset A and asset B but no muA and muB
    # TODO: need to incorporate max steps per episode
    def __init__(self,
                 w_optimal: Union[np.ndarray, list],
                 transaction_costs: Union[np.ndarray, list, float, int] = 0,
                 n_assets: Union[np.float32, np.ndarray, list] = 2,
                 mu: Union[np.float32, np.ndarray, list] = np.array([50, 200]) / 1e4,
                 sigma: Union[np.float32, np.ndarray, list] = np.array([300, 800]) / 1e4,
                 w_min: float = 0.0,
                 w_max: float = 100.0
                 ):

        # Define mean array of returns (or constant)
        self.mu = mu

        # define action parameters
        x = np.arange(-99, 100)
        self.action_space = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu)).astype(np.float32)
        self.action_space = self.action_space[self.action_space.sum(axis=1) == 0, :] / 100

        # self.action_space = spaces.Discrete(num_actions)
        # self.action_space = spaces.Box(low=action_range_min, high=action_range_max, shape=(1,), dtype=np.float32)

        # self.num_actions = num_actions

        # define state space (observation space) TODO: need to make this dynamic
        x = np.arange(1, 101)
        self.state_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu)).astype(np.float32)
        self.state_possible = self.state_possible[self.state_possible.sum(axis=1) == 100, :] / 100
        # high = np.array([w_max], dtype=np.float32)
        # low = np.array([w_min], dtype=np.float32)

        # define boundary conditions
        self.w_max = w_max
        self.w_min = w_min

        # TODO: Need to discretize the observation space to same increment as that of action space
        # self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None
        self.steps_beyond_terminated = None
        self.seed()

        # define optimal variance constant
        self.sigma = sigma
        self.sigma_mat = np.diag(self.sigma ** 2)

        # define optimal weight from min variance portfolio
        self.w_optimal = w_optimal

        # define initial amount invested

        # define transaction costs array
        self.transaction_costs = transaction_costs

        # define number of assets
        self.n_assets = n_assets

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_delta):
        """
        TODO: Need to be able to return this: next_state, reward, done, info = env.step(action)
        a good resource is this: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
        # create q table in the jupyter notebook..maybe
        https://www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/
        :param action_delta:
        :return:
        """
        err_msg = f"{action_delta!r} ({type(action_delta)}) invalid"
        assert action_delta in self.action_space, err_msg
        assert self.state is not None, "Call reset before using step method."

        # TODO: Do some calculations using current weight (need to account for other assets)
        # -> right now it is implied since 1 - A = B
        # set `new state` as the new weight due to change in market price
        # self.state += action_delta
        new_state = self.state + action_delta

        # TODO: Should we add a termination state where we hit optimal weight?
        # Maybe if weights exceed 20 and 80 percent
        terminated = True if np.any(np.any(self.state <= self.w_min) or
                                 np.any(self.state >= self.w_max) or
                                 np.any(new_state <= self.w_min) or
                                 np.any(new_state >= self.w_max)) else False

        # TODO: Calculate cost
        # # need to calculate on updated weights based on the action delta Weight
        # # then need to compare to optimal portfolio that remains constant say 50/50 is optimal
        #
        # # Calculate the Expected Utility with the Current Portfolio Weight
        # expected_utility_current = self.utility_function(self.mu, self.sigma_current)
        #
        # # Calculate the Expected Utility if the Optimal Portfolio Weight was Selected
        # expected_utility_optimal = self.utility_function(self.mu, self.sigma_optimal)
        #
        # # Calculate the Certainty Equivalent Costs in the particular period (i.e. cost of not rebalancing to Optimal
        # # Portfolio)
        # certainty_equivalent_cost = math.exp(expected_utility_optimal - expected_utility_current)
        #
        # # TODO: Calculate the Transaction Costs to be incurred if rebalancing is to take place
        # # TODO: ideally change this to linear algebra of TC.T.dot(abs(diff))
        # current_transaction_cost = 0
        # for i, asset in enumerate(range(self.n_assets)):
        #     # if first asset
        #     if i == 0:
        #         # since self.state corresponds to asset A weight (for now)
        #         current_transaction_cost += self.transaction_costs[i] * (math.fabs(self.w_optimal - self.state))
        #     # if other assets
        #     else:
        #         # TODO: need to account for multiple weights (now it considers two assets)
        #         current_transaction_cost += self.transaction_costs[i] * (math.fabs((1 - self.w_optimal) - (1 - self.state)))

        if not terminated:
            # Calculate Total Costs to be incurred if rebalancing is to take place
            # total_cost = (certainty_equivalent_cost + current_transaction_cost) * self.initial_amount_invested

            # TODO: Need to account where next_state is
            # next_state_id = np.argwhere(np.all(self.state_possible == new_state, axis=1)).item()

            # calculate reward
            reward = -expected_cost_total(w0=self.state, w1=new_state, opt_w=self.w_optimal,
                                          mu=self.mu, cov=self.sigma_mat, tc=self.transaction_costs)
        elif self.steps_beyond_terminated is None:
            # weights exceeded 0/100 % threshold
            self.steps_beyond_terminated = 0

            # Calculate Total Costs to be incurred if rebalancing is to take place
            reward = -expected_cost_total(w0=self.state, w1=new_state, opt_w=self.w_optimal,
                                          mu=self.mu, cov=self.sigma_mat, tc=self.transaction_costs)

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

        # set current state as new state
        # self.state = new_state

        return np.array(self.state, dtype=np.float32), reward, terminated

    def reset(self):
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.05, 0.05  # default low
        # )  # default high
        # self.state = np.random.uniform(low=self.w_min+0.1, high=self.w_max-0.1, size=(1,))

        # exclude one asset TODO: needs to be fixed for multi
        row_id = np.random.choice(self.state_possible.shape[0], size=1)
        self.state = self.state_possible[row_id]
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}




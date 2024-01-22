from typing import Union

import numpy as np

import gym
from gym.utils import seeding
from gym import logger

from scipy.optimize import minimize


#######################################################################################################################
# Define Necessary Functions
#######################################################################################################################


# Net Sharpe Ratio
def net_sharpe(w1, mu, cov, w0, tc):
    """

    :param w1: next state
    :param mu: mean
    :param cov: covariance diagonal matrix
    :param w0: current state
    :param tc: transaction costs
    :return: net sharpe value
    """
    return (w1.dot(mu) - cost_turnover(w0, w1, tc)) / np.sqrt(w1.dot(cov).dot(w1.T))


# Objective Function
def obj_func(x, mu, cov):
    """
    Objective Function for the Mean Variance optimization algorithm.
    :param x: tmp weight
    :param mu: mean
    :param cov: covariance diagonal matrix
    :return:
    """
    return -x.dot(mu) / np.sqrt(x.dot(cov).dot(x))
    # return 0.5 * (x.dot(cov).dot(x)) - x.dot(mu)


# Finding Optimal Weight given mean and covariance
def find_optimal_wgt(mu, cov):
    # TODO: Should we change w_max to 1 or to 2/n ? so if n = 8 limit would be 0.25 in one asset?
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
    """

    :param w0: current state weights
    :param w1: next state weights
    :param tc: transaction costs
    :return: cost turnover value
    """
    return np.sum(np.abs(w1 - w0) * tc) / 2


def expected_cost_total(w0, w1, opt_w, mu, cov, tc):
    """

    :param w0: current state weights
    :param w1: next state weights
    :param opt_w: optimal mean-variance weights
    :param mu: mean of returns
    :param cov: covariance of returns
    :param tc: transaction costs
    :return: expected cost of optimal - state net sharpe values
    """
    opt_net_sharpe = net_sharpe(w1=opt_w, mu=mu, cov=cov, w0=w0, tc=tc)
    w1_net_sharpe = net_sharpe(w1=w1, mu=mu, cov=cov, w0=w0, tc=tc)
    return opt_net_sharpe - w1_net_sharpe

class PortfolioRebalancerEnv(gym.Env):
    """
    Description:


    """
    def __init__(self,
                 w_optimal: Union[np.ndarray, list],
                 transaction_costs: Union[np.ndarray, list, float, int] = 0,
                 n_assets: Union[int, np.float32, np.ndarray, list] = 2,
                 mu: Union[np.float32, np.ndarray, list] = np.array([50, 200]) / 1e4,
                 sigma: Union[np.float32, np.ndarray, list] = np.array([300, 800]) / 1e4,
                 w_min: float = 0.0,
                 w_max: float = 100.0
                 ):

        # Define mean array of returns (or constant)
        self.mu = mu

        # define action parameters
        # x = np.arange(-99, 100)
        # self.action_space = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu)).astype(np.float32)
        # self.action_space = self.action_space[self.action_space.sum(axis=1) == 0, :] / 100
        self.action_space = np.array([0, 1])

        # self.action_space = spaces.Discrete(num_actions)
        # self.action_space = spaces.Box(low=action_range_min, high=action_range_max, shape=(1,), dtype=np.float32)

        # self.num_actions = num_actions

        # define state space (observation space) TODO: need to make this dynamic
        # x = np.arange(1, 101)
        # self.state_possible = np.array(np.meshgrid(*([x] * len(self.mu)))).T.reshape(-1, len(self.mu)).astype(
        #     np.float32)
        # self.state_possible = self.state_possible[self.state_possible.sum(axis=1) == 100, :] / 100
        # high = np.array([w_max], dtype=np.float32)
        # low = np.array([w_min], dtype=np.float32)
        # self.state = self.init_state()

        # define boundary conditions
        self.w_max = w_max
        self.w_min = w_min

        # TODO: Need to discretize the observation space to same increment as that of action space
        # self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None
        self.steps_beyond_terminated = None
        self.seed()

        # define optimal variance constant
        self.cov = sigma
        # self.sigma_mat = np.diag(self.cov ** 2)

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

    def init_state(self):
        # states from 0 to 1 and sum to 1
        state_possible = np.random.uniform(low=0, high=1, size=self.n_assets)
        state_possible /= state_possible.sum()
        state_possible = np.round(state_possible, 2)  # discretize the state space to avoid explosion

        # if not np.isclose(state_possible.sum(), 1.0, atol=1e-8):
        #     print(state_possible.sum())
        return state_possible[np.newaxis, :]

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

        # set `new state` as the new weight due to change in market price
        # self.state += action_delta
        if action_delta == 0:
            new_state = self.state.flatten()
        else:
            new_state = self.w_optimal.flatten()
        # new_state = self.state + action_delta
        # new_state = new_state.flatten()  # make 1d

        terminated = True if np.any(np.any(self.state < self.w_min) or
                                    np.any(self.state > self.w_max) or
                                    np.any(new_state < self.w_min) or
                                    np.any(new_state > self.w_max)) else False

        if not terminated:
            # Calculate Total Costs to be incurred if rebalancing is to take place
            # total_cost = (certainty_equivalent_cost + current_transaction_cost) * self.initial_amount_invested

            # next_state_id = np.argwhere(np.all(self.state_possible == new_state, axis=1)).item()

            # calculate reward
            reward = -expected_cost_total(w0=self.state, w1=new_state, opt_w=self.w_optimal,
                                          mu=self.mu, cov=self.cov, tc=self.transaction_costs)
            # reward = obj_func(x=new_state, mu=self.mu, cov=self.cov)
            # reward += cost_turnover(w0=self.state, w1=new_state, tc=self.transaction_costs)
            # reward = -reward
        elif self.steps_beyond_terminated is None:
            # weights exceeded 0/100 % threshold
            self.steps_beyond_terminated = 0

            # Calculate Total Costs to be incurred if rebalancing is to take place
            reward = -expected_cost_total(w0=self.state, w1=new_state, opt_w=self.w_optimal,
                                          mu=self.mu, cov=self.cov, tc=self.transaction_costs)
            # reward = obj_func(x=new_state, mu=self.mu, cov=self.cov)
            # reward += cost_turnover(w0=self.state, w1=new_state, tc=self.transaction_costs)
            # reward = -reward
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

        reward = np.array(reward, dtype=np.float32)
        return np.array(self.state, dtype=np.float32), reward[np.newaxis], terminated

    def reset(self):
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.05, 0.05  # default low
        # )  # default high
        # self.state = np.random.uniform(low=self.w_min+0.1, high=self.w_max-0.1, size=(1,))

        # exclude one asset TODO: needs to be fixed for multi
        # row_id = np.random.choice(self.state_possible.shape[0], size=1)
        # self.state = self.state_possible[row_id]
        self.state = self.init_state()
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}

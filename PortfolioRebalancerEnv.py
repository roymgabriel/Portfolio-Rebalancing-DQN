import math
from typing import Optional, Union

import numpy as np

import gym
from gym.utils import seeding
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class PortfolioRebalancerEnv(gym.Env):
    """
    Description:


    """

    # TODO: Will calculate optimal min variance portfolio weight outside this class
    def __init__(self,
                 mu: Union[np.float32, np.ndarray, list],
                 sigma_optimal: np.float32,
                 sigma_current: np.float32,
                 w_optimal: Union[np.ndarray, list],
                 initial_amount_invested: Union[np.float32, int],
                 transaction_costs: Union[np.ndarray, list],
                 n_assets: int,
                 action_range_min: np.float32 = -0.07,
                 action_range_max: np.float32 = 0.07,
                 num_actions: int = 15,
                 w_min: np.float32 = 0.2,
                 w_max: np.float32 = 0.8
                 ):

        # TODO: Might want to calculate var, mean outside of class

        # define action parameters
        action_range = np.linspace(action_range_min, action_range_max, num_actions)

        self.action_space = spaces.Discrete(num_actions)

        # define state space (observation space)
        high = np.array([w_max], dtype=np.float32)
        low = np.array([w_min], dtype=np.float32)

        # define boundary conditions
        self.w_max = w_max
        self.w_min = w_min

        # TODO: Need to discretize the observation space to same increment as that of action space
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None
        self.steps_beyond_terminated = None
        self.seed()

        # Define mean array of returns (or constant)
        self.mu = mu

        # define optimal variance constant
        self.sigma_optimal = sigma_optimal
        # define optimal weight from min variance portfolio
        self.w_optimal = w_optimal

        self.initial_amount_invested = initial_amount_invested

        # TODO: Need to define current mu and current sigma as mu and sigma up to current time point
        self.sigma_current = sigma_current

        # define transaction costs array
        self.transaction_costs = transaction_costs

        # define number of asssets
        self.n_assets = n_assets


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def utility_function(self, mu, sigma):
        return math.log10(1 + mu - sigma / (2 * (1 + mu) ** 2))

    def step(self, action):
        """
        TODO: Need to be able to return this: next_state, reward, done, info = env.step(action)
        a good resource is this: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
        # create q table in the jupyter notebook..maybe
        https://www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/
        :param action:
        :return:
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # TODO: Do some calculations using current weight (need to account for other assets)
        # -> right now it is implied since 1 - A = B
        # set new state as the new weight due to change in market price
        self.state += action

        # TODO: Need to define termination conditions
        # Maybe if weights exceed 20 and 80 percent
        terminated = True if any((self.state < self.w_min) or (self.state > self.w_max)) else False

        # TODO: Calculate cost
        # need to calculate on updated weights based on the action delta Weight
        # then need to compare to optimal portfolio that remains constant say 50/50 is optimal

        # Calculate the Expected Utility with the Current Portfolio Weight
        expected_utility_current = self.utility_function(self.mu, self.sigma_current)

        # Calculate the Expected Utility if the Optimal Portfolio Weight was Selected
        expected_utility_optimal = self.utility_function(self.mu, self.sigma_optimal)

        # Calculate the Certainty Equivalent Costs in the particular period (i.e. cost of not rebalancing to Optimal
        # Portfolio)
        certainty_equivalent_cost = math.exp(expected_utility_optimal - expected_utility_current) * \
                                    self.initial_amount_invested

        # TODO: Calculate the Transaction Costs to be incurred if rebalancing is to take place
        # Cost_Min['TC'][line2] = (self.CA * math.fabs(Optimal_WeightA - (Cost_Min['WeightA'][line2])) + CB * math.fabs(
        #             (1 - Optimal_WeightA) - (1 - (Cost_Min['WeightA'][line2]))))

        # current_transaction_cost = self.transaction_costs.T.dot(math.fabs(self.w_optimal - self.state))
        # TODO: ideally change this to linear algebra of TC.T.dot(abs(diff))
        current_transaction_cost = 0
        for i, asset in enumerate(self.n_assets):
            # if first asset
            if i == 0:
                # since self.state corresponds to asset A weight (for now)
                current_transaction_cost += self.transaction_costs[i] * (math.fabs(self.w_optimal - self.state))
            # if other assets
            else:
                # TODO: need to account for multiple weights (now it considers two assets)
                current_transaction_cost += self.transaction_costs[i] * (math.fabs((1 - self.w_optimal) - (1 - self.state)))




        if not terminated:
            # Calculate Total Costs to be incurred if rebalancing is to take place
            total_cost = certainty_equivalent_cost + current_transaction_cost
        elif self.steps_beyond_terminated is None:
            # weights exceeded 20/80 % threshold
            self.steps_beyond_terminated = 0

            # Calculate Total Costs to be incurred if rebalancing is to take place
            total_cost = certainty_equivalent_cost + current_transaction_cost

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

            # reset the cost
            total_cost = 0.0

        reward = - total_cost

        return np.array(self.state, dtype=np.float32), reward, terminated

    def reset(self):
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.05, 0.05  # default low
        # )  # default high
        self.state = np.random.uniform(low=self.w_min, high=self.w_max, size=(1,))
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}


    def render(self, mode="human"):
        pass

import csv
import os
from typing import Union, Optional

import gym
import torch
from gym import logger, spaces
from gym.utils import seeding

from utils import *


#######################################################################################################################
# Define Necessary Functions
#######################################################################################################################

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
        self.action_space = np.array([0, 1])  # 0 or 1 : 0 no rebalancing else rebalance for 1
        # self.action_space = spaces.Discrete(2)  # 0 or 1 : 0 no rebalancing else rebalance for 1
        # self.action_space = spaces.Box(low=0, high=action_range_max, shape=(1,), dtype=np.float32)

        # define boundary conditions
        self.w_max = w_max
        self.w_min = w_min

        self.state = None
        self.steps_beyond_terminated = None
        # self.seed()

        # define input variance constant
        self.cov = sigma

        # define optimal weight from min variance portfolio
        self.w_optimal = w_optimal

        # define transaction costs array
        self.transaction_costs = transaction_costs

        # define number of assets
        self.n_assets = n_assets

        # init state
        # self.state = self.init_state()

    def seed(self, seed=903):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def init_state(self):
        # states from 0 to 1 and sum to 1
        state_possible = np.random.uniform(low=0, high=1, size=self.n_assets)
        state_possible /= state_possible.sum()
        state_possible = np.round(state_possible, 2)  # discretize the state space to avoid explosion

        assert np.isclose(state_possible.sum(), 1.0, atol=1e-8)
        return state_possible[np.newaxis, :]

    def step(self, action):
        """
        :param action:
        :return:
        """
        # if type(action) == torch.Tensor:
        #     action = int(action.item())
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert action in self.action_space, err_msg
        assert self.state is not None, "Call reset before using step method."

        if action == 0:
            # no rebalancing
            new_state = self.state.flatten()
        elif action == 1:
            # TODO: Will need to change this for signal change
            new_state = self.w_optimal.flatten()
        else:
            raise ValueError("Wrong action value! Must be either 0 or 1!")

        # Maybe if weights exceed weight boundaries
        terminated = True if np.any(np.any(self.state <= self.w_min) or
                                    np.any(self.state >= self.w_max) or
                                    np.any(new_state <= self.w_min) or
                                    np.any(new_state >= self.w_max)) else False

        if not terminated:
            # calculate reward
            reward = -expected_cost_total(w0=self.state, w1=new_state, opt_w=self.w_optimal,
                                          mu=self.mu, cov=self.cov, tc=self.transaction_costs)
        elif self.steps_beyond_terminated is None:
            # weights exceeded 0/100 % threshold
            self.steps_beyond_terminated = 0

            # Calculate Total Costs to be incurred if rebalancing is to take place
            reward = -expected_cost_total(w0=self.state, w1=new_state, opt_w=self.w_optimal,
                                          mu=self.mu, cov=self.cov, tc=self.transaction_costs)
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

        reward = np.array(reward, dtype=np.float32)
        return np.array(self.state, dtype=np.float32), reward[np.newaxis], terminated

    def save_to_csv(self, d, csv_file):
        print("Saving states to csv...")
        # Check if the CSV file already exists
        file_exists = os.path.exists(csv_file)

        # If the file already existed, open it in append mode and add the new data
        if not file_exists:
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                # Assuming data is a 1D array; adjust accordingly for 2D arrays
                writer.writerow(d)
        else:
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                # Assuming data is a 1D array; adjust accordingly for 2D arrays
                writer.writerow(d)

    def reset(self, **kwargs):
        super().reset(seed=903)
        self.state = self.init_state()
        # self.save_to_csv(d=self.state.ravel(), csv_file=f'states_{self.n_assets}.csv')
        self.steps_beyond_terminated = None
        return np.array(self.state, dtype=np.float32), {}

import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
from scipy import stats
import scipy
import random

import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PortfolioRebalancerEnv import PortfolioRebalancerEnv



def cost_turnover(cost_vec, wgt_orig, wgt_target):
    return np.sum(np.abs(wgt_orig - wgt_target) * cost_vec)


def certainty_equivalent_ret(wgt, mu, sigma_mat):
    return np.sum(wgt * mu) - 0.5 * np.dot(wgt, np.dot(sigma_mat, wgt))


def solve_quadratic_optimization_st_linear_constraint(mu, sigma_mat):
    # Define the objective function and its gradient
    def obj_func(x, Q, c):
        return 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c, x)

    def obj_grad(x, Q, c):
        return np.dot(Q, x) + c

    # Define the constraint functions and their gradients
    def linear_constraint(x, A, b):
        return np.dot(A, x) - b

    def linear_constraint_jac(x, A, b):
        return A

    # Define the problem parameters
    Q = sigma_mat
    c = -mu
    A = np.ones_like(mu)
    b = np.array([1])

    # Define the bounds on x
    bounds = [(0, 1)] * len(mu)

    # Define the constraints
    linear_constraints = {'type': 'eq', 'fun': linear_constraint, 'jac': linear_constraint_jac, 'args': (A, b)}

    # Solve the problem
    x0 = np.ones_like(mu) / len(mu)
    result = minimize(obj_func, x0=x0, args=(Q, c), jac=obj_grad, bounds=bounds, constraints=[linear_constraints])
    return result


def cost_suboptimality(wgt_current, mu, sigma_mat):
    utility_optimal = solve_quadratic_optimization_st_linear_constraint(mu, sigma_mat)
    np.testing.assert_almost_equal(-utility_optimal.fun, certainty_equivalent_ret(utility_optimal.x, mu, sigma_mat))
    ret_ce_optimal = certainty_equivalent_ret(utility_optimal.x, mu, sigma_mat)
    ret_ce_current = certainty_equivalent_ret(wgt_current, mu, sigma_mat)
    return ret_ce_optimal - ret_ce_current


def expected_cost_total(state_wgt, action_delta, mu, sigma_mat, transaction_cost):
    cost_p1 = cost_turnover(transaction_cost, state_wgt, state_wgt + action_delta)
    cost_p2 = cost_suboptimality(state_wgt + action_delta, mu, sigma_mat)
    return cost_p1 + cost_p2


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
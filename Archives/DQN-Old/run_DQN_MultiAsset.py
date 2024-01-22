#######################################################################################################################
# This program will Define a DQN to optimally rebalance a portfolio given the number of assets with constant mean
# and covariance.
#######################################################################################################################

#######################################################################################################################
# Import necessary Libraries
#######################################################################################################################

import os
import numpy as np
import pandas as pd
import torch
from DQN_MultiAsset import DQNlearning, find_optimal_wgt

#######################################################################################################################
# Initialize Current Directory Information
#######################################################################################################################

# Define the directory name and get the path
results_directory = "DQN_MultiAsset_results"
filepath = os.getcwd() + '/' + results_directory + '/'

# Check if the directory already exists
if not os.path.exists(filepath):
    # Create the directory if it doesn't exist
    os.makedirs(filepath)

    # move into the new directory
    os.chdir(filepath)
    print(f"Directory '{filepath}' created successfully!")
else:
    # move into the already existing directory
    os.chdir(filepath)
    print(f"Directory '{filepath}' already exists!")

# for n_asset in [2, 4, 8, 32, 128]:
for n_asset in [2]:
    ##################################################################################################################
    # Initialize Necessary Constants
    ##################################################################################################################

    # mu = np.array([50, 200]) / 1e4
    # sigma = np.array([300, 800]) / 1e4
    # cov = np.diag(sigma ** 2)
    # optimal_weight = find_optimal_wgt(mu, cov)
    # x0 = np.ones(len(mu)) / len(mu)

    mu = np.linspace(50, 200, n_asset) / 1e4
    sigma = np.linspace(300, 800, n_asset) / 1e4
    cov = np.diag(sigma ** 2)
    optimal_weight = find_optimal_wgt(mu, cov)
    x0 = np.ones(len(mu)) / len(mu)

    ##################################################################################################################
    # Run DQN Algorithm
    ##################################################################################################################

    dqn_result = {}
    dqn_loss = {}
    for tc in [0, 0.0005, 0.001, 0.002]:
        dqn = DQNlearning(mu, cov, tc, gamma=0.9, min_epsilon=0.1, learning_rate=0.001)
        dqn.iterate(num_episodes=1000, max_steps_per_episode=100)
        dqn_result[tc] = dqn

    x = dqn.state_possible[:, 0]
    dqn_action_df = pd.DataFrame(index=x)
    for tc, mo in dqn_result.items():
        # in order to visualize q table
        action = []
        for j in range(mo.state_possible.shape[0]):
            qval = mo.q_network(torch.FloatTensor(mo.state_possible[j])).detach().numpy()
            action.append(mo.action_possible[qval.argmax(), 0])

        action = np.array(action)
        dqn_action_df[f"TC: {tc * 1e4:.0f} bps"] = action

    ###################################################################################################################
    # Save Results for future plotting
    ###################################################################################################################

    filename = f"dqn_actions_{n_asset}_assets.csv"
    dqn_action_df.to_csv(filename)

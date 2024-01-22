from dp_binary import *
import numpy as np
import pandas as pd
import multiprocessing


def run_dqn_for_n_assets_tc(n_asset, tc):
    ##################################################################################################################
    # Initialize Necessary Constants
    ##################################################################################################################

    mu = np.linspace(50, 200, n_asset) / 1e4
    sigma = np.linspace(300, 800, n_asset) / 1e4
    cov = np.diag(sigma ** 2)

    ##################################################################################################################
    # Run DQN Algorithm
    ##################################################################################################################

    bell = BellmanValue(mu=mu, sigma_mat=cov, transaction_cost=tc, gamma=0.9, scaling_factor=365/5, num_iters=500)
    bell.iterate()

    filename = f'results/models/dp_tc_{tc}_assets_{n_asset}'
    print("saving q table...")
    df = pd.DataFrame(bell.q_table, index=bell.state_possible[:, 0], columns=bell.action_possible)
    df.to_csv(filename)


if __name__ == "__main__":
    # List of n_asset values
    n_assets_list = [2]

    # List of transaction cost values
    # tc_list = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
    tc_list = [0, 0.0001, 0.0002, 0.0003,
               0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    # tc_list = [0.0008]

    # NOTE: Avoiding parallel running because it crashes computer

    # Create a pool of worker processes
    # num_processes = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=num_processes)

    # Generate a list of argument tuples for all combinations of n_asset and tc
    # arg_list = [(n_asset, tc) for n_asset in n_assets_list for tc in tc_list]

    # Run the DP algorithm for all combinations in parallel
    # pool.starmap(run_dqn_for_n_assets_tc, arg_list)

    # Close the pool and wait for the worker processes to finish
    # pool.close()
    # pool.join()

    # Running double for loop
    for n_asset in n_assets_list:
        for tc in tc_list:
            run_dqn_for_n_assets_tc(n_asset=n_asset, tc=tc)


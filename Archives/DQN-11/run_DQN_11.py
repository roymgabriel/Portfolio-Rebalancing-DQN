from DQN_Binary import *
import numpy as np
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

    dqn = DQNlearning(mu=mu, sigma_mat=cov, transaction_cost=tc, gamma=0.99, min_epsilon=0.1, learning_rate=0.001)
    try:
        dqn.load_models()
    except:
        print("No models exist! Training from scratch!")

    # np.random.seed(42)
    dqn.iterate(num_episodes=10000, max_steps_per_episode=100)
    dqn.save_models()


if __name__ == "__main__":
    # List of n_asset values
    n_assets_list = [2, 4, 8, 32, 128]

    # List of transaction cost values
    tc_list = [0, 0.0005, 0.001, 0.002]

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Generate a list of argument tuples for all combinations of n_asset and tc
    arg_list = [(n_asset, tc) for n_asset in n_assets_list for tc in tc_list]

    # Run the DQN algorithm for all combinations in parallel
    pool.starmap(run_dqn_for_n_assets_tc, arg_list)

    # Close the pool and wait for the worker processes to finish
    pool.close()
    pool.join()

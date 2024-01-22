from DQN_Binary import *
import multiprocessing


def run_training(n_asset, tc):
    # RL:
    BATCH_SIZE = 128
    GAMMA = 0.9
    EPS_START = 0.9999
    EPS_END = 0.05
    EPS_DECAY = 1_000
    TAU = 0.005
    LR = 1e-4
    mem_cap = 10_000
    mode = 'FF'

    num_episodes = 20_000

    mu = np.linspace(50, 200, n_asset) / 1e4
    sigma = np.linspace(300, 800, n_asset) / 1e4
    cov = np.diag(sigma ** 2)

    agent = Agent(mu=mu, cov=cov, batch_size=BATCH_SIZE, gamma=GAMMA,
                  eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, tau=TAU, lr=LR,
                  tc=tc, mem_cap=mem_cap, mode=mode)

    # try:
    #     agent.load_models()
    # except Exception as e:
    #     print(f"Error loading models: {e}")
    #     print("Training from scratch!")

    agent.iterate(num_episodes=num_episodes)


if __name__ == "__main__":
    # List of n_asset values
    n_assets_list = [2]

    # List of transaction cost values
    tc_list = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]

    # Create a pool of worker processes
    # num_processes = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=num_processes)
    #
    # # Generate a list of argument tuples for all combinations of n_asset and tc
    # arg_list = [(n_asset, tc) for n_asset in n_assets_list for tc in tc_list]
    #
    # # Run the DQN algorithm for all combinations in parallel
    # pool.starmap(run_training, arg_list)
    #
    # # Close the pool and wait for the worker processes to finish
    # pool.close()
    # pool.join()
    for n_asset in n_assets_list:
        for tc in tc_list:
            run_training(n_asset=n_asset, tc=tc)

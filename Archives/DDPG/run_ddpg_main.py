from DDPG import *
import numpy as np
from utils import plotLearning
import multiprocessing


def run_dqn_for_n_assets(n_asset):
    ##################################################################################################################
    # Initialize Necessary Constants
    ##################################################################################################################

    mu = np.linspace(50, 200, n_asset) / 1e4
    sigma = np.linspace(300, 800, n_asset) / 1e4
    cov = np.diag(sigma ** 2)

    ##################################################################################################################
    # Run DQN Algorithm
    ##################################################################################################################

    rewards_dict = {}
    for tc in [0, 0.0005, 0.001, 0.002]:

        env = ENV(mu=mu, sigma_mat=cov, transaction_cost=tc)
        agent = Agent(alpha=0.01, beta=0.01, input_dims=[n_asset], tau=0.01, tc=tc,
                      batch_size=100000, layer1_size=400, layer2_size=300, n_actions=n_asset)

        try:
            agent.load_models()
        except:
            print("No models exist! Training from scratch!")
        np.random.seed(42)

        score_history = []
        for i in range(100000):
            current_state = env.init_state()
            done = False
            score = 0
            while not done:
                current_action = agent.choose_action(current_state)
                new_state, reward, done = env.step(current_action=current_action, current_state=current_state)
                agent.remember(current_state, current_action, reward, new_state, int(done))
                agent.learn()
                score += reward
                current_state = new_state

            score_history.append(score)

            if i % 100 == 0:
                agent.save_models()

            print('num_assets = ', n_asset, 'episode ', i, 'score %.2f' % score,
                  'trailing 1000 episodes avg %.3f' % np.mean(score_history[-1000:]))

        rewards_dict[tc] = score_history

    filename = f"results/plots/ddpg_reward_{n_asset}_assets.png"
    plotLearning(reward_dict=rewards_dict, filename=filename, window=1000)
    rewards_dict = {}


if __name__ == "__main__":
    # List of n_asset values
    n_assets_list = [2, 4, 8, 32, 128]

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Run the DQN algorithm for each n_asset value in parallel
    pool.map(run_dqn_for_n_assets, n_assets_list)

    # Close the pool and wait for the worker processes to finish
    pool.close()
    pool.join()

#######################################################################################################################
# This program will Define a DQN to optimally rebalance a portfolio given the number of assets with constant mean
# and covariance.
#######################################################################################################################

#######################################################################################################################
# Import necessary Libraries
#######################################################################################################################

from DDPG import *
import numpy as np
from utils import plotLearning
import multiprocessing

#######################################################################################################################
# Initialize Current Directory Information
#######################################################################################################################

# # Define the directory name and get the path
# results_directory = "DDPG_MultiAsset_results"
# filepath = os.getcwd() + '/' + results_directory + '/'
#
# # Check if the directory already exists
# if not os.path.exists(filepath):
#     # Create the directory if it doesn't exist
#     os.makedirs(filepath)
#
#     # move into the new directory
#     os.chdir(filepath)
#     print(f"Directory '{filepath}' created successfully!")
# else:
#     # move into the already existing directory
#     os.chdir(filepath)
#     print(f"Directory '{filepath}' already exists!")

# for n_asset in [2, 4, 8, 32, 128]:
for n_asset in [2, 4, 8, 32, 128]:
    ##################################################################################################################
    # Initialize Necessary Constants
    ##################################################################################################################

    mu = np.linspace(50, 200, n_asset) / 1e4
    sigma = np.linspace(300, 800, n_asset) / 1e4
    cov = np.diag(sigma ** 2)
    optimal_weight = find_optimal_wgt(mu, cov)
    x0 = np.ones(len(mu)) / len(mu)

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

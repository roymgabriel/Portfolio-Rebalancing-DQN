from DDPG import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_asset = 2
x = np.arange(1, 101)
state_possible = np.array(np.meshgrid(*([x] * n_asset))).T.reshape(-1, n_asset).astype(np.float32)
state_possible = state_possible[state_possible.sum(axis=1) == 100, :] / 100

x = state_possible[:, 0]
ddpg_action_df = pd.DataFrame(index=x)
for tc in [0, 0.0005, 0.001, 0.002]:
    agent = Agent(alpha=0.01, beta=0.01, input_dims=[n_asset], tau=0.01, tc=tc,
                  batch_size=100000, layer1_size=400, layer2_size=300, n_actions=n_asset)
    agent.load_models()
    action = []
    for current_state in state_possible:
        tmp_action = agent.choose_action(current_state)
        action.append(tmp_action[0])  # just store action of first asset

    action = np.array(action)
    ddpg_action_df[f"TC: {tc * 1e4:.0f} bps"] = action

print(ddpg_action_df)

mu = np.linspace(50, 200, n_asset) / 1e4
sigma = np.linspace(300, 800, n_asset) / 1e4
cov = np.diag(sigma ** 2)
x0 = np.ones(len(mu)) / len(mu)

optimal_weight = find_optimal_wgt(mu=mu, cov=cov)
print("optimal weight: ", optimal_weight)

plt.rcParams.update({"font.size": 18})
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ddpg_action_df.plot(ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], linestyle="dotted")
ax.set_xlabel("Weight on First Asset")
ax.set_ylabel("Suggested Delta Weight on Asset 1")
ax.axvline(optimal_weight[0], color="red", linestyle="dotted")
ax.axhline(0, color="red", linestyle="dotted")
ax.legend()
plt.tight_layout()
plt.show()

filepath = "results/plots/2asset.png"
plt.savefig(filepath)
plt.close()

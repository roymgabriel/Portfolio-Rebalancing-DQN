import matplotlib.pyplot as plt
import pandas as pd

def plotLearning(reward_dict, filename, window=5):
    df = pd.DataFrame(reward_dict)
    plot_df = pd.DataFrame()
    for tc in df.columns:
        plot_df[f"TC: {tc * 1e4:.0f} bps"] = df[tc].rolling(window=window).mean()

    ax = plot_df.plot(colormap='jet', linestyle="dotted", title='Training Reward over Time.')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    plt.legend()
    plt.savefig(filename)



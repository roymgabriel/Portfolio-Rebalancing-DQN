# Portfolio-Rebalancing-DQN

## DP Algorithm

Two DP algorithms were implemented; one done by [Roy](/DP_Roy) and the other by [Pratyush](/DP_Pratyush/). There are also binary implementations of the DP algorithm and they can be found with suffix `_Binary`: [DP_Binary](/DP_Binary) and [DP_Binary_split](/DP_Binary_split/). If the suffix `_split` exists in the name of the directory, it means the objective function changes based on the simulation run in the files [Policy Comparison Notebook (zoomed in transaction costs)](PolicyComparison_10.ipynb) or [Policy Comparison Notebook (regular)](PolicyComparison_EC.ipynb).

## DQN Algorithm

This can be seen under the files [DQN_Binary](/DQN_Binary/) and [DQN_Binary_split](/DQN_Binary_split/), where `_split` denotes that the objective function changes based on the function you are trying to anaylze in the simulation. If you are looking for a continuous action space implementation of DQN, please see the file [DQN-Old](/Archives/DQN-Old/).

## Analyzing Results

As mentioned previously, the notebooks [Policy Comparison Notebook (zoomed in transaction costs)](PolicyComparison_10.ipynb) or [Policy Comparison Notebook (regular)](PolicyComparison_EC.ipynb) run a simulation against other policies and splits the objective function into gains (Sharpe ratio) and losses (cost turnover) to analyze each of the policies. They both have the same structure except the notebook ending in `_10` zoomes in on low transaction costs between `0` and `10` bps.

If you are interested in looking for optimal rebalance zone (or equivalently zone of no rebalance), please see the notebooks [VisualizeResults_Split](VisualizeResults_Split.ipynb) and [VisualizeResults](VisualizeResults.ipynb) for the simulation and for the general algorithm respectively.


## Arhives

In the archives, you will find old implementations of the algorithm, as well as a stab at different models like [DDPG](Archives/DDPG) (continuous action and state space), [DQN-11](Archives/DQN-11) (multi-period implementation), and an implementation by Dr. Yang [DP-Rebal-Prof](Archives/DP-Rebal-Prof), which also includes a stab at [signal change](Archives/DP-Rebal-Prof/signal_change.py).

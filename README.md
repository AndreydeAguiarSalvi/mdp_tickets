# README #

This is my implementation of Lottery Tickets Hyphotesis using Markov Decision Process to choose the connections to prune.
[[Paper of Lottery Tickets Hypothesis - first version]](https://arxiv.org/pdf/1803.03635.pdf)
[[Paper of Lottery Tickets Hypothesis - second version]](https://arxiv.org/pdf/1903.01611.pdf)

### What is this repository for? ###

* Work for the discipline Automated Planning, 2019-2, Teacher [Felipe Meneguzzi](https://github.com/meneguzzi) at Pontifícia Universidade Católica do Rio Grande do Sul (PUCRS)
* Pruning Multi-Layer Perceptrons
* Classical Planning

## Installation
### Environment - Prerequisites
* Anaconda 3
* Nvidia-driver
Create your anaconda environment and install:
```
conda install -c anaconda pandas 
conda install -c anaconda numpy 
conda install pytorch torchvision cudatoolkit=YOUR_NVIDIA-DRIVER_VERSION -c pytorch
```
### Folders ###

#### common ####
Here we have the implementations of:
* Agent
* State
* Neural Network (net.py)
* Some utility functions (utils.py)

#### data ####
The dataset is downloaded here.

#### policy_maker ####
Here we have the implementations of:
* epsilon_search: the Decreasing Epsilon-Greedy Search to learn the Q-Table.
** prunner.py: create the agent and learn the Q-Table.
** params.json: the hyperparameters for the Neural Network, Agent and etc.
** job.sh: to run manny tests in bash.

#### results ####
The results will be saved here.
* analyse_results: will summarize the results obtained.
* test.py: will analyse the results of your model on test dataset.

#### training ####
Just train a model without pruning.

#### training_pruning ####
Will train and prune iteratively:
* lottery_tickets.py: prune with the original algorithm from Frankle and Carbin.
* mdp_tickets: prune with a Q-Table already learned.
* params.json: the hyperparameters to use.
* job.sh: to execute manny tests in bash.

## Credit
```
@misc{frankle2018lottery,
    title={The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks},
    author={Jonathan Frankle and Michael Carbin},
    year={2018},
    eprint={1803.03635},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
@misc{frankle2019stabilizing,
    title={Stabilizing the Lottery Ticket Hypothesis},
    author={Jonathan Frankle and Gintare Karolina Dziugaite and Daniel M. Roy and Michael Carbin},
    year={2019},
    eprint={1903.01611},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
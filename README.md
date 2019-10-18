# README #

This is my implementation of Lottery Tickets Hyphotesis using Markov Decision Process to choose the connections to prune.
[[Paper of Lottery Tickets Hypothesis - first version]](https://arxiv.org/pdf/1803.03635.pdf)
[[Paper of Lottery Tickets Hypothesis - second version]](https://arxiv.org/pdf/1903.01611.pdf)

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

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
### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

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
# FoolsGold
A sybil-resilient distributed learning protocol that penalizes Sybils based on their gradient similarity.

### FoolsGold is also described in two papers:

 1. Peer-reviewed conference paper [pdf](https://www.cs.ubc.ca/~bestchai/papers/foolsgold-raid2020.pdf):
```
"The Limitations of Federated Learning in Sybil Settings." 
Clement Fung, Chris J.M. Yoon, Ivan Beschastnikh.
To appear in 23rd International Symposium on Research in Attacks, Intrusions and Defenses (RAID) 2020.
```
Bibtex:
```
@InProceedings{Fung2020,
  title     = {{The Limitations of Federated Learning in Sybil Settings}},
  author    = {Clement Fung and Chris J. M. Yoon and Ivan Beschastnikh},
  year      = {2020},
  series    = {RAID},
  booktitle = {Symposium on Research in Attacks, Intrusion, and Defenses},
}
```
 2. [Arxiv paper](https://arxiv.org/abs/1808.04866)

## Running a minimal MNIST example

### Get the MNIST data. 
Download and gunzip files from http://yann.lecun.com/exdb/mnist/  
Move all the outputted files to `ML/data/mnist`.
Navigate to that directory: `cd ML/data/mnist` and run `parse_mnist.py`

### Create poisoned MNIST 1-7 data
From main directory navigate to the ML directory: `cd ML/`  
Run: `python code/misslabel_dataset.py mnist 1 7` 

### Run FoolsGold
From main directory navigate to the ML directory: `cd ML/`  
And run the following command for a 5 sybil, 1-7 attack on mnist.
```
python code/ML_main.py mnist 1000 5_1_7
```

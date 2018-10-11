# FoolsGold
A sybil-resilient distributed learning protocol that penalizes Sybils based on their gradient similarity.

## Running a minimal MNIST example

### Get the MNIST data. 
Download and gunzip files from http://yann.lecun.com/exdb/mnist/  
Move all the outputted files to `ML/data/mnist`.
Navigate to that directory: `cd ML/data/mnist` and run `parse_mnist.py`

### Create poisoned MNIST 1-7 data
Navigate to the ML directory: `cd ML/`  
Run: `python code/misslabel_dataset.py mnist 1 7` 

### Run FoolsGold
Navigate to the ML directory: `cd ML/`  
And run the following command for a 5 sybil, 1-7 attack on mnist.
```
python code/ML_main.py mnist 1000 5_1_7
```

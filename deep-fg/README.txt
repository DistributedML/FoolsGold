# Deep FoolsGold
Implementation of FoolsGold with deep learning in PyTorch. 

## Installation
Option 1. Create a new conda environment using 
```
conda env create -f dfgenv.yml
```
Option 2. Install individual dependencies
```
conda install -c conda-forge pyhocon 
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install -c conda-forge matplotlib 
pip install torchnet
```



## Data
1. Download VGGFace2 Dataset from http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
2. Preprocess data using datasets.process_vgg.py
or
1. Extract data.tar.xz in data/vggface2

## Running
Federated training is implemented in fg.trainer.FedTrainer
FoolsGold is implemented in fg.foolsgold

1. Change configuration in train.hocon
2. ```fed_train.py train.hocon```

import os
import shutil

from pyhocon import ConfigFactory


class Option(object):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)

        # ------------- general options ----------------------------------------
        self.save_path = self.conf['save_path']  # log path
        self.data_path = self.conf['data_path']  # path for loading data set
        self.dataset = self.conf['dataset']  # options: imagenet | cifar10
        self.seed = self.conf['seed']  # manually set RNG seed
        self.gpu = self.conf['gpu']  # GPU id to use, e.g. "0,1,2,3"
        self.print_freq = self.conf['print_freq']
        # ------------- data options -------------------------------------------
        self.num_workers = self.conf['num_workers']  # number of threads used for data loading

        # ------------- common optimization options ----------------------------
        self.opt = self.conf['opt'] # optimizer (sgd, adam)
        self.batch_size = self.conf['batch_size']  # mini-batch size
        self.desired_bs = self.conf['desired_bs']  # mini-batch size
        self.momentum = self.conf['momentum']  # momentum
        self.weight_decay = self.conf['weight_decay']  # weight decay
        self.lr = self.conf['lr']   # learning rate
        self.epochs = self.conf['epochs']   # learning rate
        self.lr_step = self.conf['lr_step'] # epochs to reduce lr

        # ------------- model options ------------------------------------------
        self.arch_type = self.conf['arch_type']  # options: resnet | preresnet | vgg
        self.arch_depth = self.conf['arch_depth']
        self.experiment_id = self.conf['experiment_id']  # identifier for experiment
        self.n_classes = self.conf['n_classes']  # number of classes in the dataset
        self.pretrained = self.conf['pretrained']     

        # ---------- resume or retrain options ---------------------------------
        # path to model to retrain with, load model state_dict only
        self.retrain = None if len(self.conf['retrain']) == 0 else self.conf['retrain']
        # path to directory containing checkpoint, load state_dicts of model and optimizer, as well as training epoch
        self.resume = None if len(self.conf['resume']) == 0 else self.conf['resume']

        # ---------- foolsgold options -------------------------------------------
        self.num_sybils = self.conf['num_sybils']
        self.num_sybil_groups = self.conf['num_sybil_groups']
        self.use_fg = self.conf['use_fg']
        self.use_memory = self.conf['use_memory']

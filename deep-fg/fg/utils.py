import matplotlib.pyplot as plt
import numpy as np
import pickle
import torchnet as tnt
import pdb 
import torch 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class IncrementVisdomLineLogger(object):
    def __init__(self, opts=None):
        self.line_logger = tnt.logger.VisdomPlotLogger('line', opts=opts)
        self.count = 0
    
    def log(self, y, name):
        self.line_logger.log(self.count, y, name=name)
        self.count += 1

# returns tnt.ConfusionMeter
class ConfusionMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.pred = torch.tensor([]).long()
        self.target = torch.tensor([]).long()
    
    # pred: (N)
    # target: (N)
    def add(self, pred, target):
        try:
            self.pred = torch.cat((self.pred, torch.tensor(pred)), 0)
            self.target = torch.cat((self.target, target.cpu()), 0)
        except:
            pdb.set_trace()
    
    def value(self):
        try:
            confusion_meter = tnt.meter.ConfusionMeter(self.num_classes, normalized=True)
            confusion_meter.add(self.pred, self.target)
        except:
            pdb.set_trace()
        return confusion_meter.value()


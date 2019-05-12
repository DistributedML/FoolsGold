import argparse
import utils
from utils import AverageMeter
from utils import accuracy
from utils import IncrementVisdomLineLogger
import pdb
from checkpoint import Checkpoint
import foolsgold as fg

import time
import numpy as np
import torch
from torch.autograd import Variable
import torchnet as tnt
import os
import sklearn.metrics.pairwise as smp

"""
Given model, optimizer, criterion, and loader, trains the model
"""
class BaseTrainer(object):
    def __init__(self, option, model, train_loader, val_loader, test_loader, optimizer, criterion):
        self.option = option
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion

        self.epoch_loss_plotter = tnt.logger.VisdomPlotLogger('line', opts={'title': 'Epoch Loss', 'xlabel':"Epochs", 'ylabel':"Loss"})
        self.batch_loss_plotter = IncrementVisdomLineLogger(opts={'title': 'Batch Loss', 'xlabel':"Batch", 'ylabel':"Loss"})

        self.checkpoint = Checkpoint(option)
        self.best_top1 = 0
        self.start_epoch = 0
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.option.resume:
            checkpoint = self.checkpoint.load_checkpoint()
            if checkpoint is None:
                return
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.best_top1 = checkpoint['best_top1']
            self.start_epoch = checkpoint['epoch']

    def update_lr(self, epoch):
        gamma = 0
        for step in self.option.lr_step:
            if epoch + 1.0 > int(step):
                gamma += 1
        lr = self.option.lr * (0.1 ** gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print("Training with lr: {}".format(lr))

    def train(self):
        for epoch in range(self.start_epoch, self.option.epochs):
            self.update_lr(epoch)

            train_loss = self.train_iter(epoch)
            val_loss, top1, attack_rate = self.validate()
            self.epoch_loss_plotter.log(epoch, train_loss, name="train")
            self.epoch_loss_plotter.log(epoch, val_loss, name="val")
            
            # save checkpoint
            is_best = top1 > self.best_top1
            if is_best:
                self.best_top1 = top1
            save_state = {
                'epoch': epoch+1,
                'model': self.model,
                'optimizer': self.optimizer,
                'top1': top1,
                'best_top1': self.best_top1
            }
            self.checkpoint.save_checkpoint(save_state, is_best)


    def train_iter(self, epoch):
        batch_time = AverageMeter() # Time it takes to complete one desired bs
        data_time = AverageMeter()  # Time it takes to load data
        losses = AverageMeter() # Accumulates for the whole epoch
        print_freq_loss = AverageMeter() # Reset every print freq
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        self.optimizer.zero_grad()
        batch_count = 0
        for i, (input, target) in enumerate(self.train_loader):   
                       
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda()

            # compute output
            output = self.model(input)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            print_freq_loss.update(loss.item(), input.size(0))

            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            loss.backward()
            # compute gradient and do SGD step after accumulating gradients
            if i % (self.option.desired_bs // self.option.batch_size) == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_count += 1

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_count % self.option.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
                    self.batch_loss_plotter.log(print_freq_loss.avg, name="train")
                    print_freq_loss = AverageMeter()
                
        return losses.avg

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = self.model(input)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.option.print_freq == 0:
                print('Val: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(self.val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

        return losses.avg, top1.avg


class FedTrainer(object):
    def __init__(self, option, model, train_loader, val_loader, test_loader, optimizer, criterion, client_loaders, sybil_loaders, iidness=[.0, .0]):
        self.option = option
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.iidness = iidness

        self.epoch_loss_plotter = tnt.logger.VisdomPlotLogger('line', opts={'title': 'Epoch Loss', 'xlabel':"Epochs", 'ylabel':"Loss"})
        self.batch_loss_plotter = IncrementVisdomLineLogger(opts={'title': 'Batch Loss', 'xlabel':"Batch", 'ylabel':"Loss"})
        self.train_confusion_plotter = tnt.logger.VisdomLogger('heatmap', opts={'title': 'Train Confusion matrix',
                                                                'columnnames': list(range(option.n_classes)),
                                                                'rownames': list(range(option.n_classes))})
        self.val_confusion_plotter = tnt.logger.VisdomLogger('heatmap', opts={'title': 'Val Confusion matrix',
                                                                'columnnames': list(range(option.n_classes)),
                                                                'rownames': list(range(option.n_classes))})

        self.memory = None
        self.wv_history = []
        self.client_loaders = client_loaders
        self.sybil_loaders = sybil_loaders

        self.checkpoint = Checkpoint(option)
        self.best_top1 = 0
        self.start_epoch = 0
        self._load_checkpoint()

    def _load_checkpoint(self):
        if self.option.resume:
            checkpoint = self.checkpoint.load_checkpoint()
            if checkpoint is None:
                return
            self.model = checkpoint['model']
            self.optimizer = checkpoint['optimizer']
            self.best_top1 = checkpoint['best_top1']
            self.start_epoch = checkpoint['epoch']

    def update_lr(self, epoch):
        gamma = 0
        for step in self.option.lr_step:
            if epoch + 1.0 > int(step):
                gamma += 1
        lr = self.option.lr * (0.1 ** gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print("Training with lr: {}".format(lr))

    def train(self):
        # in the fed learning case, epoch is analogous to iter
        best_loss = float('inf')
        for epoch in range(self.start_epoch, self.option.epochs):
            # self.update_lr(epoch)

            train_loss = self.train_iter(epoch)
            print("Epoch {}/{}\t Train Loss: {}".format(epoch, self.option.epochs, train_loss))
            self.batch_loss_plotter.log(train_loss, name="train")

            # val_loss, top1 = self.validate()
            # self.epoch_loss_plotter.log(epoch, train_loss, name="train")
            # self.epoch_loss_plotter.log(epoch, val_loss, name="val")
            
            if epoch % 10 == 0:
                val_loss, top1, attack_rate = self.validate()
                print("Epoch {}/{}\t Train Loss: {}\t Val Loss: {}\t Attack Rate: {}".format(
                    epoch, self.option.epochs, train_loss, val_loss, attack_rate))

                # save checkpoint
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                save_state = {
                    'epoch': epoch+1,
                    'model': self.model,
                    'optimizer': self.optimizer,
                    'top1': top1,
                    'best_top1': self.best_top1,
                    'attack_rate': attack_rate
                }
                self.checkpoint.save_checkpoint(save_state, is_best)


    # for each client
    #   calculate gradient for one iter
    #   store gradients
    #   zero gradients

    # Note: the batchnorm statistics are automatically updated in our fake fed learning
    def train_iter(self, epoch):
        self.model.train()
        client_losses = []
        preds = []
        targets = []   
        confusion_meter = utils.ConfusionMeter(self.option.n_classes)

        all_loaders = self.client_loaders + self.sybil_loaders     
        # Compute gradients from all the clients
        client_grads = []
        for client_loader in all_loaders:
            self.optimizer.zero_grad()
            input, target = next(iter(client_loader))
            input = input.cuda()
            target = target.cuda()
            output = self.model(input)
            loss = self.criterion(output, target)
            loss.backward()            

            # Store statistics
            client_losses.append(loss.item())
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()[0].tolist()
            preds.extend(pred)
            targets.extend(target.tolist())

            client_grad = []
            for name, params in self.model.named_parameters():
                if params.requires_grad:
                    client_grad.append(params.grad.cpu().clone())
            client_grads.append(client_grad)
                
        # Update model
        # Add all the gradients to the model gradient
        self.optimizer.zero_grad()
        agg_grads = self.aggregate_gradients(client_grads)
        for i, (name, params) in enumerate(self.model.named_parameters()):
            if params.requires_grad:
                params.grad = agg_grads[i].cuda()

        confusion_meter.add(preds, torch.tensor(targets))
        self.train_confusion_plotter.log(confusion_meter.value())
        # Update model
        self.optimizer.step()
        return np.array(client_losses).mean()

    def aggregate_gradients(self, client_grads):
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((num_clients, grad_len))
        
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory += grads

        if self.option.use_fg:
            if self.option.use_memory:
                wv = fg.foolsgold(self.memory) # Use FG
            else:
                wv = fg.foolsgold(grads) # Use FG
        else:
            # wv = fg.foolsgold(grads) # Use FG w/o memory
            wv = np.ones(num_clients) # Don't use FG
        print(wv)
        self.wv_history.append(wv)

        agg_grads = []  
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i]
            temp = temp / len(client_grads)
            agg_grads.append(temp)
            
        return agg_grads

    def validate(self, test=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        confusion_meter = utils.ConfusionMeter(self.option.n_classes)
        # switch to evaluate mode
        self.model.eval()

        preds = []
        targets = []

        end = time.time()
        loader = self.test_loader if test else self.val_loader
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = self.model(input)
            loss = self.criterion(output, target)            

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()[0].tolist()

            preds.extend(pred)
            targets.extend(target.tolist())

            confusion_meter.add(pred, target)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        # Compute attack rate 0 to 1
        # Percentage of True 0's classified as 1's
        preds = np.array(preds)
        targets = np.array(targets)
        n_poisoned = (preds[targets == 0] == 1).sum() # Number of true 0's classified as 1's
        n_total = (targets == 0).sum() # Number of true 0's
        attack_rate = n_poisoned / n_total

        print(' Val Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        self.val_confusion_plotter.log(confusion_meter.value())
        return losses.avg, top1.avg, attack_rate
    
    # Save val_acc, attack_rate (how many 0's are classified as 1), wv_history, memory
    def save_state(self):
        val_loss, top1, attack_rate = self.validate()
        state = {
            "val_loss": val_loss,
            "val_acc": top1.item(),
            "attack_rate": attack_rate,
            "wv_history": self.wv_history,
            "memory": self.memory
        }
        # TODO: Refactor out
        save_dir = os.path.join(self.option.save_path, "iidness")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "{}-{}.pth".format(int(100*self.iidness[0]), int(100*self.iidness[1])))

        torch.save(state, save_path)
        return state
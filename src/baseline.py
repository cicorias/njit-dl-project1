#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from contextlib import redirect_stdout
from copy import deepcopy

import argparse
import matplotlib.pyplot as plt
from typing import Iterable, Set, Tuple, Union
import datetime as dt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable

from continuum import ClassIncremental
from continuum.datasets import Core50
from continuum.tasks import split_train_val
from torchvision.transforms.transforms import Normalize, ToTensor


def redirect(text, path='./out.txt', *args, **kwargs):
    # first send to normal stdout.
    print(text, *args, **kwargs)
    with open(path, 'a+') as out:
        with redirect_stdout(out):
            print(text, *args, **kwargs)

def on_task_update(task_id, x_mem, y_mem):
    """
    EWC weight updater
    """
    pass

def train(classifier, task_id, train_loader, criterion, optimizer, max_epochs, convergence_criterion):
    def print2(parms, *aargs, **kwargs):
        redirect(parms, path=args.outfile, *aargs, **kwargs)

    # End early criterion
    last_avg_running_loss = convergence_criterion #  TODO: not used currently
    did_converge = False

    for epoch in range(max_epochs):

        # End if the loss has converged to criterion
        if did_converge:
            break
            
        print2(f"<------ Epoch {epoch + 1} ------->")

        running_loss = 0.0
        train_total = 0.0
        train_correct = 0.0 
        for i, (x, y, t) in enumerate(train_loader):

            # Outputs batches of data, one scenario at a time
            x, y = x.cuda(), y.cuda()
            outputs = classifier(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print training statistics
            running_loss += loss.item()
            train_total += y.size(0)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == y).sum().item()
            
            if i % 100 == 99:
                avg_running_loss = running_loss / 3200
                print2(f'[Mini-batch {i + 1}] avg loss: {avg_running_loss:.5f}')
                # End early criterion
                if avg_running_loss < convergence_criterion:
                    did_converge = True
                    break
                last_avg_running_loss = avg_running_loss
                running_loss = 0.0
                        
        print2(f"Training accuracy: {100.0 * train_correct / train_total}%")
    return


###### EWC Stuff ######

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset, scenario, task_id):

        self.model = model
        self.dataset = deepcopy(dataset)
        self.scenario = scenario
        self.task_id = task_id

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()

        replay_examples = taskset_with_replay(self.scenario, self.task_id, 0.1)

        # overwrite taskset examples with previously seen examples
        self.dataset._x = replay_examples['x']
        self.dataset._y = replay_examples['y']
        self.dataset._t = replay_examples['t']

        ewc_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        for i, (x, y, t) in enumerate(ewc_loader):
            x, y = x.cuda(), y.cuda()
            outputs = self.model(x)
            _, train_predicted = torch.max(outputs.data, 1)
            loss = nn.functional.nll_loss(nn.functional.log_softmax(outputs, dim=1), train_predicted)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset._y)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        num_params = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
            num_params += 1
        return loss / num_params


def train_ewc(classifier, task_id, train_loader, criterion, ewc, importance, optimizer, max_epochs, convergence_criterion):
    def print2(parms, *aargs, **kwargs):
        redirect(parms, path=args.outfile, *aargs, **kwargs)

    print2("Training with EWC")

    # End early criterion
    last_avg_running_loss = convergence_criterion #  TODO: not used currently
    did_converge = False

    for epoch in range(max_epochs):

        # End if the loss has converged to criterion
        if did_converge:
            break
            
        print2(f"<------ Epoch {epoch + 1} ------->")

        running_loss = 0.0
        train_total = 0.0
        train_correct = 0.0 
        for i, (x, y, t) in enumerate(train_loader):

            # Outputs batches of data, one scenario at a time
            x, y = x.cuda(), y.cuda()
            outputs = classifier(x)
            loss = criterion(outputs, y) + importance * ewc.penalty(classifier)
            loss.backward()
            optimizer.step()

            # print training statistics
            running_loss += loss.item()
            train_total += y.size(0)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == y).sum().item()
            
            if i % 100 == 99:
                avg_running_loss = running_loss / 3200
                print2(f'[Mini-batch {i + 1}] avg loss: {avg_running_loss:.5f}')
                # End early criterion
                if avg_running_loss < convergence_criterion:
                    did_converge = True
                    break
                last_avg_running_loss = avg_running_loss
                running_loss = 0.0
                        
        print2(f"Training accuracy: {100.0 * train_correct / train_total}%")
    return

###### END EWC STUFF ########

# Continuous learning via Rehearsal 
def taskset_with_replay(scenario, task_id, proportion):
    replay_examples = {
        'x': np.array([], dtype='<U49'),
        'y': np.array([], dtype='int64'),
        't': np.array([], dtype='int64')
    }

    # Grab new random replay examples from each of the previous tasks
    for prev_id, prev_taskset in enumerate(scenario):
        if prev_id == task_id:
            break

        sz = round(len(prev_taskset) * proportion)
        replay_examples['x'] = np.append(
            replay_examples['x'], np.random.choice(prev_taskset._x, size=sz, replace=False)
        )
        replay_examples['y'] = np.append(
            replay_examples['y'], np.random.choice(prev_taskset._y, size=sz, replace=False)
        )
        replay_examples['t'] = np.append(
            replay_examples['t'], np.random.choice(prev_taskset._t, size=sz, replace=False)
        )

    return replay_examples


def main(args):
    def print2(parms, *aargs, **kwargs):
        redirect(parms, path=args.outfile, *aargs, **kwargs)

    start_time = time.time()

    # print args recap
    print2(args, end='\n\n')
    print2('hello {}'.format('world'))
    
    # Load the core50 data
    # TODO: check the symbolic links as for me no '../' prefix needed.

    if args.download:
        print2('cli switch download set to True so download will occur...')
        print2('  alternatively the batch script fetch_data_and_setup.sh can be used')

    
    print2('using directory for data_path path {}'.format(args.data_path))


    core50 = Core50(args.data_path, train=True, download=args.download)
    core50_val = Core50(args.data_path, train=False, download=args.download)

    # A new classes scenario, using continuum
    scenario = ClassIncremental(
        core50,
        increment=5,
        initial_increment=10,
        transformations=[ ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    )
    scenario_val = ClassIncremental(
        core50_val,
        increment=5,
        initial_increment=10,
        transformations=[ ToTensor(), Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    )

    print2(f"Number of classes: {scenario.nb_classes}.")
    print2(f"Number of tasks: {scenario.nb_tasks}.")

    # Define a model
    # model
    if args.classifier == 'resnet18':
        classifier = models.resnet18(pretrained=True)
        classifier.fc = torch.nn.Linear(512, args.n_classes)
    
    elif args.classifier == 'resnet101':
        classifier = models.resnet101(pretrained=True)
        classifier.fc = nn.Linear(2048, args.n_classes)

    elif args.classifier == 'resnet34':
        classifier = models.resnet34(pretrained=True)
        classifier.fc = nn.Linear(512, args.n_classes)
    
    else:
        raise Exception('no classifier picked')

    # Fix for RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
    if torch.cuda.is_available():
        classifier.cuda()

    # Tune the model hyperparameters
    max_epochs = args.epochs # 8
    convergence_criterion = args.convergence_criterion # 0.004  # End early if loss is less than this
    lr = args.lr  # 0.00001
    weight_decay = args.weight_decay # 0.000001
    momentum = args.momentum # 0.9

    # Define a loss function and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        classifier.parameters(), 
        lr=lr, 
        weight_decay=weight_decay, 
        momentum=momentum
        )
    print2("Criterion: " + str(criterion))
    print2("Optimizer: " + str(optimizer))

    # Validation accuracies
    accuracies = []

    # Iterate through our NC scenario
    for task_id, train_taskset in enumerate(scenario):

        print2(f"<-------------- Task {task_id + 1} ---------------->")

        # Use replay if it's specified
        if args.replay:

            # Add replay examples to current taskset
            replay_examples = taskset_with_replay(scenario, task_id, args.replay)
            train_taskset._x = np.append(train_taskset._x, replay_examples['x'])
            train_taskset._y = np.append(train_taskset._y, replay_examples['y'])
            train_taskset._t = np.append(train_taskset._t, replay_examples['t'])

        train_loader = DataLoader(train_taskset, batch_size=32, shuffle=True)
        unq_cls_train = np.unique(train_taskset._y)

        print2(f"This task contains {len(unq_cls_train)} unique classes")
        print2(f"Training classes: {unq_cls_train}")

        # Train the model
        classifier.train()
        if args.importance:
            # EWC
            if task_id == 0:
                train(classifier, task_id, train_loader, criterion, optimizer, max_epochs, convergence_criterion)
            else:
                old_tasks = []
                for prev_id, prev_taskset in enumerate(scenario):
                    if prev_id == task_id:
                        break
                    else:
                        old_tasks = old_tasks + list(prev_taskset._x)
                train_ewc(classifier, task_id, train_loader, criterion, EWC(classifier, train_taskset, scenario, task_id), args.importance, optimizer, max_epochs, convergence_criterion)
        else:
            train(classifier, task_id, train_loader, criterion, optimizer, max_epochs, convergence_criterion)

        print2("Finished Training")
        classifier.eval()

        # Validate against separate validation data
        cum_accuracy = 0.0
        for val_task_id, val_taskset in enumerate(scenario_val):

            # Validate on all previously trained tasks (but not future tasks)
            # if val_task_id > task_id:
            #     break

            val_loader = DataLoader(val_taskset, batch_size=32, shuffle=True)

            # Make sure we're validating the correct classes
            unq_cls_validate = np.unique(val_taskset._y)
            print2(f"Validating classes: {unq_cls_validate}")

            total = 0.0
            correct = 0.0
            pred_classes = np.array([])
            with torch.no_grad():
                for x, y, t in val_loader:
                    x, y = x.cuda(), y.cuda()
                    outputs = classifier(x)
                    _, predicted = torch.max(outputs.data, 1)
                    pred_classes = np.unique(np.append(pred_classes, predicted.cpu()))
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            
            print2(f"Classes predicted: {pred_classes}")
            print2(f"Validation Accuracy: {100.0 * correct / total}%")
            cum_accuracy += (correct / total)
        
        print2(f"Average Accuracy: {cum_accuracy / 9}")
        accuracies.append((cum_accuracy / 9))   
        print2(f"Average Accuracy: {100.0 * cum_accuracy / 9.0}%")

        
    
    # Running Time
    print2("--- %s seconds ---" % (time.time() - start_time))

    # TO DO Add EWC Training

    # Some plots over time
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], accuracies, '-o', label="Naive")
    #plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], rehe_accs, '-o', label="Rehearsal")
    #plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], ewc_accs, '-o', label="EWC")
    plt.xlabel('Tasks Encountered', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)
    plt.title('Rehersal Strategy on Core50 w/ResNet18', fontsize=14)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.legend(prop={'size': 16})
    plt.show()
    filenames = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig('continuum/output/run_'+filenames+'.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Ted David Shawn - NJIT')

    parser.add_argument('--data_path', type=str, default='core50/data/')

    parser.add_argument('--download', type=bool, default=False)

    ### Use these command line args to set parameters for the model ###

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='resnet18',
                        choices=['resnet18', 'resnet101', 'resnet34'])

    # Optimization
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')

    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')

    parser.add_argument('--weight_decay', type=float, default=0.000001,
                        help='weight decay')

    parser.add_argument('--convergence_criterion', type=float, default=0.004 ,
                        help='convergence_criterion ')

    parser.add_argument('--momentum', type=float, default=0.8,
                        help='momentum')

    parser.add_argument('--replay', type=float, default=0.0, help='proportion of training to replay')

    parser.add_argument('--importance', type=int, default=0.1, help='EWC importance criterion')

    import datetime
    temp_out_file = datetime.datetime.now().strftime('./%Y_%m_%d-%H_%M_%S') + '.txt'
    parser.add_argument('--outfile', type=str, default=temp_out_file)

    args = parser.parse_args()
    
    # Core 50 uses 50 classes
    args.n_classes = 50

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    if args.cuda:
        print('cuda IS available')
    else:
        print('cuda / GPU not available.')


    main(args)

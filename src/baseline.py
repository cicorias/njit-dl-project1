#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from contextlib import redirect_stdout

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

def train_ewc(model, device, task_id, x_train, y_train, optimizer, epoch):
    """
    EWC Trainer
    """
    pass

def taskset_with_replay(scenario, task_id, train_taskset, proportion):
    replay_examples = {
        'x': np.array([], dtype='<U49'),
        'y': np.array([], dtype='int64'),
        't': np.array([], dtype='int64')
    }

    for prev_id, prev_taskset in enumerate(scenario):
        if prev_id == task_id:
            break

        sz = round(len(prev_taskset) * proportion)
        # Grab new replay examples
        replay_examples['x'] = np.append(
            replay_examples['x'], np.random.choice(prev_taskset._x, size=sz, replace=False)
        )
        replay_examples['y'] = np.append(
            replay_examples['y'], np.random.choice(prev_taskset._y, size=sz, replace=False)
        )
        replay_examples['t'] = np.append(
            replay_examples['t'], np.random.choice(prev_taskset._t, size=sz, replace=False)
        )

        # Add replay examples
        train_taskset._x = np.append(train_taskset._x, replay_examples['x'])
        train_taskset._y = np.append(train_taskset._y, replay_examples['y'])
        train_taskset._t = np.append(train_taskset._t, replay_examples['t'])

    return train_taskset


def main(args):
    def print2(parms, *aargs, **kwargs):
        redirect(parms, path=args.outfile, *aargs, **kwargs)
    

    start_time = time.time()

    # print args recap
    print2(args, end='\n\n')
    print2('hello {}'.format('world'))
    
    # Load the core50 data
    # TODO: check the symbolic links as for me no '../' prefix needed.
    core50 = Core50("core50/data/", train=True, download=False)
    core50_val = Core50("core50/data", train=False, download=False)

    # A new classes scenario
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
    momentum = args.momentum # 0.9 #  TODO: not used currently

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

    # Naive_acc
    naive_accs = []
    # replay_examples = {
    #     'x': np.array([], dtype='<U49'),
    #     'y': np.array([], dtype='int64'),
    #     't': np.array([], dtype='int64')
    # }

    # Iterate through our NC scenario
    for task_id, train_taskset in enumerate(scenario):

        print2(f"<-------------- Task {task_id + 1} ---------------->")

        # Use replay if it's specified
        if args.replay:
            train_taskset = taskset_with_replay(scenario, task_id, train_taskset, args.replay)

        train_loader = DataLoader(train_taskset, batch_size=32, shuffle=True)
        unq_cls_train = np.unique(train_taskset._y)

        print2(f"This task contains {len(unq_cls_train)} unique classes")
        print2(f"Training classes: {unq_cls_train}")

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
        naive_accs.append((cum_accuracy / 9))   
        print2(f"Average Accuracy: {100.0 * cum_accuracy / 9.0}%")

        classifier.train()
    
    # Running Time

    print2("--- %s seconds ---" % (time.time() - start_time))

    # TO DO Add EWC Training

    # Plot

    
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], naive_accs, '-o', label="Rehersal")
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

    # General
    # parser.add_argument('--scenario', type=str, default="multi-task-nc",
    #                     choices=['ni', 'multi-task-nc', 'nic'])
    # parser.add_argument('--preload_data', type=bool, default=True,
    #                     help='preload data into RAM')

    # Model
    parser.add_argument('-cls', '--classifier', type=str, default='resnet18',
                        choices=['resnet18', 'resnet101'])

    # Optimization
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')

    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs')

    parser.add_argument('--weight_decay', type=float, default=0.000001,
                        help='weight decay')

    parser.add_argument('--convergence_criterion', type=float, default=0.004 ,
                        help='convergence_criterion ')

    parser.add_argument('--momentum', type=float, default=0.8,
                        help='momentum')

    parser.add_argument('--replay', type=float, default=0.15, help='proportion of training to replay')

    import datetime
    temp_out_file = datetime.datetime.now().strftime('./%Y_%m_%d-%H_%M_%S') + '.txt'
    parser.add_argument('--outfile', type=str, default=temp_out_file)
           

# TODO: fix these as parms.
# add and replace
#     max_epochs = 8
#     convergence_criterion = 0.004  # End early if loss is less than this
#     lr = 0.00001
#     weight_decay = 0.000001
#     momentum = 0.9 #  TODO: not used currently

    # Continual Learning
    # parser.add_argument('--replay_examples', type=int, default=0,
    #                     help='data examples to keep in memory for each batch '
    #                          'for replay.')

    args = parser.parse_args()
    
    args.n_classes = 50
    #args.input_size = [3, 128, 128]

    args.cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.cuda else 'cpu'

    if args.cuda:
        print('cuda IS available')
    else:
        print('cuda / GPU not available.')


    main(args)

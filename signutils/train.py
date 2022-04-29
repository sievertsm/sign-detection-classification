import os
from glob import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import seaborn as sns

from PIL import Image

import time
import copy

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F


def get_elapsed_time(start_time, unit='hr'):
    '''Helper function to determine the elapsed time
    '''
    current_time = time.time()  
    elapsed_time = current_time - start_time
    if unit=='hr':
        elapsed_time/=3600
    elif unit=='min':
        elapsed_time/=60
    elif unit=='sec':
        pass
    return elapsed_time


def format_training_records(loss_record=None, acc_record=None):
    '''Helper function to format record dictionaries into dataframes
    '''
    
    df_loss_epoch, df_loss_batch, df_acc = None, None, None
    
    if loss_record:
        df_loss_epoch = pd.DataFrame({phase: loss_record[phase]['epoch'] for phase in ['train', 'val']})
        df_loss_batch = pd.DataFrame({'train': loss_record['train']['batch']})
    
    if acc_record:
        df_acc = pd.DataFrame(acc_record)
        
    return df_loss_epoch, df_loss_batch, df_acc


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=5, scheduler=None, print_every=100):
    '''Function to train classification model. Modified from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    Input: 
    model         --
    criterion     --
    optimizer     --
    dataloaders   --
    dataset_sizes --
    device        --
    num_epochs    --
    scheduler     --
    print_every   --

    Output:
    model       --
    loss_record --
    acc_record  --
    '''

    model.to(device=device)

    #initialize recording variables
    best_model_wts = copy.deepcopy(model.state_dict()) # store initial weights of the model
    best_acc = 0.0 # initialize best accuracy to 0.0
    loss_record = {'train': {'batch': [], 'epoch': []}, 'val': {'batch': [], 'epoch': []}} # dictionary for storing loss
    acc_record = {'train': [], 'val': []} # dictionary for storing accuracy

    # record time when training was started
    since = time.time()

    # loop over all epochs
    for epoch in range(num_epochs):

        # print beginning of epoch
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 70)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # set model to appropriate mode
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # intialize running record variables for the epoch
            running_loss = 0.0
            running_corrects = 0

            # Iterate over each batch of data
            for b, sample in enumerate(dataloaders[phase]):
                # unpack sample
                inputs, labels = sample 
                # move variables onto device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # record batch loss
                loss_record[phase]['batch'].append(loss.item())
                
                # print update
                if phase=='train' and b%print_every == 0:
                    elapsed_time = get_elapsed_time(since)
                    print(f"Batch: {b:04}, loss: {loss.item():.5f} time: {round(elapsed_time, 3):.3f}")
            
            # update learning rate with the scheduler
            if not scheduler:
                pass
            else:
                if phase == 'train':
                    scheduler.step()

            # compute loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # record epoch loss & accuracy
            loss_record[phase]['epoch'].append(epoch_loss)
            acc_record[phase].append(epoch_acc.item())

            elapsed_time = get_elapsed_time(since)
            print(f'Epoch: {epoch:04}, Phase: {phase:5}, Loss: {epoch_loss:.5f}, Acc: {epoch_acc:.5f}, Time: {round(elapsed_time, 3):.3f}')

            # deep copy the model if validation accuracy has improved
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # final print statements
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_record, acc_record
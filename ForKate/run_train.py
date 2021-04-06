from comet_ml import Experiment


# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="0EKSrlH9OVngYfgQCrauwqLEt",
                        project_name="mri-interpretation", workspace="polina")
import argparse
import pickle
import time

import numpy as np
import os
import sys
import functools
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from datetime import datetime
import torch
import torch.nn
import torch.utils.data as data_utils
import torch.optim as optim
import torch.utils.data as torch_data
import torch.nn.functional as F
# from torchsummary import summary
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torchio
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import *
import random
import nilearn
from nilearn import plotting
%matplotlib inline

from model import MriNetGrad
from data import HCP_MRI, MriData

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_X', help='path_to_X', type=str, default= '/data/adni/tensors_cut.npz')
parser.add_argument('--path_to_labels', help='path_to_labels', type=str, default='/data/adni/labels.npy')
parser.add_argument('-b', '--batch_size', default=10,
                    type=int, help='Batch size for training')
parser.add_argument('--gpu0', default=0,
                    type=int, help='Use cuda1 to train model')
parser.add_argument('--gpu1', default=1,
                    type=int, help='Use cuda1 to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-5, type=float, help='initial learning rate')
parser.add_argument('-epoch', '--epoch', default=30,
                    type=int, help='max epoch for training')
parser.add_argument('--save_folder', default='/checkpoints',
                    help='Location to save checkpoint models')
parser.add_argument('--weight_decay', default=1e-5,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--momentum', default=0.999, type=float, help='momentum')
parser.add_argument('--betas', default=0.5,
                    type=float)
parser.add_argument('--load', default=False, help='resume net for retraining')
args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
sys.stdout.flush()

CHECKPOINTS_DIR = args.save_folder
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)


def setup_experiment(title, logdir="./tb"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S")).replace(":","_")
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    best_model_path = f"{title}.best.pth"
    return writer, experiment_name, best_model_path

def get_absmax(dataset):
    absmax = 0.
    for (img, target) in dataset:
        img = torch.FloatTensor(img).to(device)
        absmax = max(absmax, img.abs().max().item())
        del img, target
    return absmax

def AbsMaxScale(img, absmax):
    return img / absmax

##LOAD DATA

# hcp_dataset = HCP_MRI(
#     paths= PATH_TO_MRI,
#     target_path= behavioral_path,
#     load_online=True
# )
# hcp_absmax = 435.0126647949219 # get_absmax(la5_dataset)
# hcp_dataset.transform = functools.partial(AbsMaxScale, absmax=hcp_absmax)

# transform = CropOrPad(
#      (180, 180, 180))
# hcp_dataset.transform = transform\
    
# print('Loading Dataset...')

# ##MODEL
# torch.manual_seed(1)
# np.random.seed(1)

# c = 32
# model = MriNetGrad(c)

# sys.stdout.flush()


def get_metric(net, data_loader):
    net.eval()
    correct = 0
    true = []
    labels = []
    for data, target in tqdm(data_loader):
        data = data.to(device,dtype=torch.float)
        target = target.to(device)

        out = net(data)
        pred = out.data.max(1)[1] # get the index of the max log-probability
        true.append(pred.data.cpu().numpy())
        labels.append(target)
        correct += pred.eq(target.data).cpu().sum()
        del data, target, out, pred
    accuracy = 100. * correct / len(data_loader.dataset)
    true = np.concatenate(true)
    roc_auc = roc_auc_score(labels, true)
    pr = precision_score(labels,  true)
    rec = recall_score(labels, true)

    return accuracy.item(), roc_auc, pr, rec

def get_accuracy(net, data_loader):
    net.eval()
    correct = 0
    for data, target in tqdm(data_loader):
        data = data.to(device,dtype=torch.float)
        target = target.to(device)

        out = net(data)
        pred = out.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        del data, target
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()

def get_loss(net, data_loader):
    net.eval()
    loss = 0 
    correct = 0
    for data, target in tqdm(data_loader):
        data = data.to(device, dtype=torch.float)
        target = target.to(device)
    
        out = net(data)
        loss += criterion(out, target).item()*len(data)
        pred = out.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

        del data, target, out 
    accuracy = 100. * correct / len(data_loader.dataset)
    return loss / len(data_loader.dataset), accuracy.item()


def train(epochs, net, criterion, optimizer, train_loader, val_loader, scheduler=None, verbose=True, save=False, experiment= False):
    best_val_loss = 100000 #100_000
    best_val_acc = 0
    best_model = None
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    loss, acc = get_loss(net, train_loader)
    train_loss_list.append(loss)
    train_acc_list.append(acc)
    loss, acc = get_loss(net, val_loader)
    val_loss_list.append(loss)
    val_acc_list.append(acc)
    del loss, acc
    if verbose:
        print('Epoch {:02d}/{} || Loss:  Train {:.4f} | Validation {:.4f}'.format(0, epochs, train_loss_list[-1], val_loss_list[-1]))

    net.to(device)
    for epoch in range(1, epochs+1):
        net.train()
        train_loss = 0
        train_correct = 0
        for X, y in train_loader:
            # Perform one step of minibatch stochastic gradient descent
            X, y = X.to(device, dtype=torch.float), y.to(device)
            optimizer.zero_grad()
#             print(type(X))
            out = net(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(X)
            pred = out.data.max(1)[1] # get the index of the max log-probability
            train_correct += pred.eq(y.data).cpu().sum()
            del X, y, out, loss #freeing gpu space
        accuracy = 100. * train_correct / len(train_loader.dataset)
        train_loss_list.append(train_loss/len(train_loader.dataset))
        train_acc_list.append(accuracy.item())
            
        
        # define NN evaluation, i.e. turn off dropouts, batchnorms, etc.
        loss, acc = get_loss(net, val_loader)
        val_loss_list.append(loss)
        val_acc_list.append(acc)
        
        if scheduler is not None:
            scheduler.step(val_acc_list[-1])

        if save and val_acc_list[-1] > best_val_acc:
            torch.save(net.state_dict(), CHECKPOINTS_DIR + 'best_acc_model_' + model_name)    
            best_val_acc = val_acc_list[-1]
            
        if save and val_loss_list[-1] < best_val_loss:
            torch.save(net.state_dict(), CHECKPOINTS_DIR + 'best_val_loss_model_' + model_name)
            best_val_loss = val_loss_list[-1]
            
        if save and epoch%5==0:
            torch.save(net.state_dict(), CHECKPOINTS_DIR + str(epoch) + '_epoch_model_' + model_name)
            
        freq = 1
        if verbose and epoch%freq==0:
            print('Epoch {:02d}/{} || Loss:  Train {:.4f} | Validation {:.4f} Acc: Train {:.4f} | Validation {:.4f}'.format(epoch, epochs, train_loss_list[-1], val_loss_list[-1], train_acc_list[-1], val_acc_list[-1] ))
        if experiment:
                experiment.log_metric("train_loss", train_loss_list[-1])
                experiment.log_metric("validate_loss", val_loss_list[-1])
                experiment.log_metric("train_acc", train_acc_list[-1])
                experiment.log_metric("validate_acc", val_acc_list[-1])
                experiment.log_epoch_end(epoch)
    return train_loss_list, val_loss_list, train_acc_list, val_acc_list    



if __name__ == '__main__':
    args = parser.parse_args()

    X = np.load(args.path_to_X)
    y = np.load(args.path_to_labels)
    X = X.f.arr_0
    print('Loading Dataset...')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_vall_acc_list = []
    cross_vall_roc_auc_list = []
    j = 0
    torch.manual_seed(82)
    torch.cuda.manual_seed(82)
    np.random.seed(82)
    for train_index, test_index in skf.split(X,y):
        print('Doing {} split'.format(j))
        j += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_dataset = MriData(X_train, y_train)
        test_dataset = MriData(X_test, y_test)
        
        model_name = '3DCNN_Freesurfer_cv5_{}'.format(j)
        experiment.set_name("3DCNN_Freesurfer_cv5_{}".format(j))
        torch.save(X_train, CHECKPOINTS_DIR + 'train_X' +model_name)
        torch.save(X_test, CHECKPOINTS_DIR + 'val_X' +model_name)
        torch.save(y_train, CHECKPOINTS_DIR + 'train_y' +model_name)
        torch.save(y_test, CHECKPOINTS_DIR + 'val_y' +model_name)
        del X_train, X_test, y_train, y_test
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 
        
        c = 32
        model = MriNetGrad(c)
    #     model.to(device)
        if torch.cuda.device_count() > 1:
            d_ids= [args.gpu0,args.gpu1]
            print("Let's use", d_ids, "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model, device_ids=d_ids)
            model.to(device)
       
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_deca)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        EPOCHS = args.epoch

        train_loss_list, val_loss_list, train_acc_list, val_acc_list = train(EPOCHS, model, criterion, optimizer, train_loader, val_loader, scheduler=scheduler, save=True, experiment= experiment) 
        acc, roc_auc, pr, rec =  get_metric(model, val_loader)
        cross_vall_acc_list.append(acc)
        cross_vall_roc_auc_list.apeend(roc_auc)
        print(cross_vall_acc_list[-1], cross_vall_roc_auc_list[-1], pr, rec)
    print('Mean cross-validation accuracy (5-folds):', np.mean(cross_vall_acc_list))
    print('Std cross-validation accuracy (5-folds):', np.std(cross_vall_acc_list, ddof=1))
    print('Mean cross-validation roc_auc (5-folds):', np.mean(cross_vall_roc_auc_list))
    print('Std cross-validation roc_auc (5-folds):', np.std(cross_vall_roc_auc_list, ddof=1))
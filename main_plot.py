#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)

if(0): # plot for MNIST convex 
    numusers = 5
    num_glob_iters = 800
    dataset = "Mnist"
    local_ep = [20,20,20,20]
    lamda = [15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005]
    beta =  [1.0, 1.0, 0.001, 1.0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.1,0.1,0.1,0.1]
    algorithms = [ "pFedMe_p","pFedMe","PerAvg_p","FedAvg"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(1): # plot for Synthetic covex
    numusers = 10
    num_glob_iters = 200
    dataset = "Cifar10"#"Synthetic"
    local_ep = [20,20,20,20]
    lamda = [15,15,15,15]#[20,20,20,20]
    learning_rate = [0.01, 0.01,0.01,0.01]#[0.005, 0.005, 0.005, 0.005]
    beta =  [0.01, 0.01,1.0,1.0]#[1.0, 1.0, 0.001, 1.0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5]
    personal_learning_rate = [0.01,0.001,0.01,0.01]

    algorithms = ["pFedMe"] #[FedMe_p","pFedMe","FedAvg","PerAvg_p"]"FedAvg","FedSRWADMM",
    plot_summary_one_figure_cifar10_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
    # plot_summary_one_figure_mnist_Beta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
    #                            learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k=K, personal_learning_rate = personal_learning_rate)

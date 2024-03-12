
# -*- coding: utf-8 -*-

"""
Created on Jul 20 17:35:24 2022


@author: HUI_ZHANG
E-mail : 202234949@mail.sdu.edu.cn  OR  bigserendipty@gmail.com
Article: "Mini-Batch Forward Automatic Differentiation based Efficient Adaptive Optimization Algorithm for TSK Fuzzy System"

This code is the first version of the above article.
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original article, 
   please contact the authors of related article.
"""


import torch
import numpy
from configs import *
from dataloader import read_dataset
from utils import loss_plot
from model import TSK_FS
from MBFAD_EAO import MBFAD_EAO


# fix random seeds for reproducibility.
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = False
numpy.random.seed(SEED)

# run MBFAD-EAO algorithm to optimize TSK-FS.
def run(cfg):

    train_x, train_y, test_x, test_y = read_dataset(cfg.path, cfg.v)  # load dataset and split it into training set and testing set.
    dataset = {"train feature": train_x, "train label": train_y, "test feature": test_x, "test label": test_y,}
    model = TSK_FS(M = cfg.M, nMFs = cfg.nMFs, P = cfg.P)  # Instantiate a TSK-FS model.
    train_loss, test_loss, time_loss = MBFAD_EAO(model = model, dataset = dataset, config = cfg)  # use MBFAD-EAO algorithm to optimize TSK-FS.

    return train_loss, test_loss, time_loss


# entrance of program.
if __name__ == "__main__":
    
    cfg = config("PM10")  # configuration class
    train_loss, test_loss, time_loss = run(cfg)  # run this program.
    loss_plot(train_loss, test_loss, time_loss, cfg.dataset_name, "MBFAD_EAO")  # plot result.


"""
run-gqn.py

Script to train the a GQN on the Shepard-Metzler dataset
in accordance to the hyperparameter settings described in
the supplementary materials of the paper.
"""
import random
import math
from argparse import ArgumentParser

# Torch
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# TensorboardX
from tensorboardX import SummaryWriter

from gqn import GenerativeQueryNetwork, partition, Annealer
from shepardmetzler import ShepardMetzler
#from placeholder import PlaceholderData as ShepardMetzler

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
#print(cuda)
#print(device)
# Random seeding
random.seed(99)
torch.manual_seed(99)
if cuda: torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model
    

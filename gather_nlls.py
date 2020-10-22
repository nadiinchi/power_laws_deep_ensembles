import pandas as pd
import numpy as np
import os
from my_utils import get_size, compute_k
import my_utils
import pickle
import pandas as pd
import data
import argparse
import sys
import logger
from gather_nlls_module import ComputeNLLs

parser = argparse.ArgumentParser(description='Run a lot of experiments')
parser.add_argument('--dataset', type=int, default=100, metavar='DATASET', required=False,
                    help='dataset: 100 | 10')
parser.add_argument('--model', type=str, default="VGG16", metavar='MODEL', required=False,
                    help='model name: WideResNet or VGG16')
parser.add_argument('--setting', type=str, default="reg", metavar='MODEL', required=False,
                    help='setting: reg or noreg')
parser.add_argument('--setup', type=int, default=1, metavar='1/2', required=False,
                    help='setup')
parser.add_argument('--reg', type=str, default="optimal", metavar='1/2', required=False,
                    help='optimal | grid')
parser.add_argument('--comment', type=str, default="", metavar='T', help='comment to the experiment')

args = parser.parse_args()

exp_label = "setup%d_%s_CIFAR%d_%s_%s%s"%(args.setup, args.reg,\
                                          args.dataset, args.model, args.setting, args.comment)
log = logger.Logger(exp_label, base="gather_logs/")
log.print(args)

if args.model == "VGG16":
    if args.dataset == 100:
        if args.setting == "noreg":
            logdirs = ["logs/oct/train.py-CIFAR100_VGG16/width_noreg_ens/",\
               "logs/jan200/train.py-CIFAR100_VGG16/width_forstd/"]
            temps = np.arange(0.25, 8.25, 0.25)
        else:
            logdirs = ["logs/oct/train.py-CIFAR100_VGG16/width_ens/",\
               "logs/jan200/train.py-CIFAR100_VGG16/width_reg_forstd/",\
               "logs/may/train.py-CIFAR100_VGG16/width_reg_forstd/"]
            
            temps = np.arange(0.5, 3.2, 0.05)
    else: # 10
        if args.setting == "noreg":
            llogdirs = ["logs/oct/train.py-CIFAR10_VGG16/width_noreg_ens/",\
               "logs/jan200/train.py-CIFAR10_VGG16/width_forstd/"]
            temps = np.arange(0.25, 8.25, 0.25)
        else:
            logdirs = ["logs/oct/train.py-CIFAR10_VGG16/width_ens/",\
               "logs/jan200/train.py-CIFAR10_VGG16/width_reg_forstd/",\
               "logs/jun/train.py-CIFAR10_VGG16/width_reg/"]
            temps = np.arange(0.5, 3.2, 0.05) 
else: # WideResNet
    if args.dataset == 100:
        if args.setting == "reg":
            logdirs = ["logs/apr200/train.py-CIFAR100_WideResNet28x10/width_ens/"]
            temps = np.arange(0.5, 1.55, 0.01) 
    if args.dataset == 10:
        if args.setting == "reg":
            logdirs = ["logs/apr200/train.py-CIFAR10_WideResNet28x10/width_ens/"]
            temps = np.arange(0.5, 2, 0.01) 
               
computer = ComputeNLLs(setup=args.setup, regime=args.reg, temps=temps, dir=log.path)
log.print(computer.compute_nlls(logdirs, args.model, args.dataset, args.setting, log,\
                            plen=1, reverse_order=False, max_std=10**5, max_enslen=10**5))




The code for the paper 
[**On Power Laws in Deep Ensembles**](https://arxiv.org/abs/2007.08483), NeurIPS'20 

#### Main scripts:

__train.py__: A script for training (and evaluating) the ensembles of VGG and WideResNet of different network sizes and ensemble sizes, for several times (e. g. for averaging the results). The script is based on [this repository](https://github.com/timgaripov/swa).

Training VGG on CIFAR100/CIFAR10:
```(bash)
python3 train.py --dir=logs/oct --data_path=./data/ --dataset=CIFAR100 --use_test --transform=VGG --model=VGG16 --save_freq=200 --print_freq=5 --epochs=200 --wd=0.001 --lr=0.05 --dropout 0.5 --comment width64 --seed 25477 --width 64 --num-nets 8 --num-exps=5 --not-save-weights
```

Training WideResNet on CIFAR100/CIFAR10:
```(bash)
python3 train.py --dir=logs/oct --data_path=./data/ --dataset=CIFAR100 --use_test --transform=ResNet --model=WideResNet28x10 --save_freq=200 --print_freq=5 --epochs=200 --wd=0.0003 --lr=0.1 --dropout 0.0 --comment width160 --seed 25477 --width 160 --num-nets 8 --num-exps=5 --not-save-weights
```
Parameters:
* --dir: where to save logs / models
* --data_path: a dir to the data (if not exist, the data will be downloaded ito this directory)
* --dataset: CIFAR100 / CIFAR10
* --model: VGG16 / WideResNet28x10
* --width: width factor (to vary network sizes); options are listed in [consts.py](https://github.com/nadiinchi/power_laws_deep_ensembles/blob/main/hypers.py)
* --num-nets: number of networks to train
* --num-exps: number of ensembles to train
* --wd, --lr, --dropout: hyperparameters, listed in [consts.py](https://github.com/nadiinchi/power_laws_deep_ensembles/blob/main/hypers.py) for different width factors for VGG and WideResNet
* --comment: additional string to be used in the name of the folder containing the results of the run
* --epochs: number of trainign epochs (we always use 200)
* --use_test: if specified, test set is used, otherwise validation set (a part of training set) is used; needed for tuning hyperparameters
* --transform: data transformation and augmentation to use (VGG / ResNet); should be specified according to the model chosen
* --save_freq / print_freq: how frequently to save the model / log
* --not-save-weights: if specified, the weights are not saved (useful when training huge ensembles)

__gather_nlls.py__: A script for computing non-calibrated NLLs / NLLs with fixed temperature / CNLLs of trained ensembles.

Usage:
```(bash)
python3 gather_nlls.py --dataset CIFAR100 --model VGG16 --setup 1 --reg optimal --comment _my_results
```

Parameters:
* --dataset: CIFAR100 / CIFAR10
* --model: VGG16 / WideResNet28x10
* --setup: 1 - applying temperature after averaging, 2 - applying temperature before averaging
* --reg: optimal - minimizing w.r.t. temperature; grid - computing NLL for each temperature on the grid
* --comment: additional string to be used in the name of the file containing the results of the run

#### Contents
* train.py: training/evaluation script
* consts.py: hyperparameters for all considered models and their sizes
* data.py: data processing code
* metrics.py: functions to compute NLL, CNLL etc. of a single model (borrowed from [this repo](https://github.com/bayesgroup/pytorch-ensembles))
* models.py: architectures specifications (standard PyTorch implementation)
* my_utils.py, utils.py: technical stuff
* logger.py: logging functions (author: @senya-ashukha)
* gather_nlls.py: script for computing NLLS / CNLLS
* gather_nlls_module.py: code for processing a large bunch of trained ensembles and computing NLLs/CNLLs for different network sizes, ensemble sizes and runs

#### Attribution

Parts of this code are based on the following repositories:
- [Stochastic Weight Averaging (SWA)](https://github.com/timgaripov/swa). Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson.
- [Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning](https://github.com/bayesgroup/pytorch-ensembles). Arsenii Ashukha, Alexander Lyzhov, Dmitry Molchanov, Dmitry Vetrov.
- [PyTorch](https://github.com/pytorch/pytorch) (we used version 1.2.0)

#### Citation

```
@article{lobacheva20power,
  title={On Power Laws in Deep Ensembles},
  author={Lobacheva, Ekaterina and Chirkova, Nadezhda and Kodryan, Maxim and Vetrov, Dmitry}
  journal={In Proceedings of the Neural Information Processing Systems (NeurIPS'20), Vancouver, Canada},
  year={2020}
}
```

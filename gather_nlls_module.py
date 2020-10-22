import pandas as pd
import numpy as np
import os
from my_utils import get_size, compute_k
import my_utils
import pickle
import pandas as pd
import data

from sklearn.model_selection import KFold
from collections import defaultdict
from scipy.optimize import minimize

def softmax(x):
    e_x = np.exp(x - np.reshape(np.max(x,axis=1),(-1,1)))
    return e_x / np.reshape(e_x.sum(axis=1),(-1,1))

def log_softmax(x):
    e_x = np.exp(x - np.reshape(np.max(x,axis=1),(-1,1)))
    return np.log(e_x / np.reshape(e_x.sum(axis=1),(-1,1)))

def get_ll(log_preds, targets, **args):
    return log_preds[np.arange(len(targets)), targets].mean()

def apply_t(log_preds, t):
     return log_softmax(log_preds / t)
    
def ts1(log_preds, targets):
    f = lambda t: -get_ll(apply_t(log_preds, t), targets)
    res = minimize(f, 1, method='nelder-mead', options={'xtol': 1e-3})
    return res.x[0]

def get_nll_setup2(list_log_preds, targets, t):
    preds = np.concatenate([np.exp(apply_t(log_preds, t))[:, :, None] \
                                for log_preds in list_log_preds], axis=2)
    predictions = np.mean(preds, axis=2)
    return -get_ll(np.log(predictions), targets)

def ts2(list_log_preds, targets):
    f = lambda t: get_nll_setup2(list_log_preds, targets, t)    
    res = minimize(f, 1, method='nelder-mead', options={'xtol': 1e-2})
    return res.x[0]

def metrics_kfold1(
        log_preds, targets, n_splits=2, n_runs=5, verbose=False, temp_scale=False, **args):
    # log_preds is np array with log preds
    temps = []
    if temp_scale:
        nll = 0
        for runs in range(n_runs):
            for i, (tr_idx, te_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).\
                                                 split(log_preds)):
                train_t = ts1(log_preds[tr_idx], targets[tr_idx])
                test_lp = apply_t(log_preds[te_idx], train_t)
                temps.append(train_t)
                v = -get_ll(test_lp, targets[te_idx])
                nll += v/(n_splits*n_runs)
    else:
        nll = -get_ll(log_preds, targets)

    return nll, temps

def metrics_kfold2(
        list_log_preds, targets, n_splits=2, n_runs=5, verbose=False, temp_scale=False, **args):
    # list_log_preds is a list of np.arrays with log preds
    temps = []
    if temp_scale:
        nll = 0
        for runs in range(n_runs):
            for i, (tr_idx, te_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).\
                                                 split(list_log_preds[0])):
                train_t = ts2([log_preds[tr_idx] for log_preds in list_log_preds], \
                               targets[tr_idx])
                v = get_nll_setup2([log_preds[te_idx] for log_preds in list_log_preds], \
                                   targets[te_idx], train_t)
                temps.append(train_t)
                nll += v/(n_splits*n_runs)
    else: # not used and do not use
        nll = get_nll_setup2(list_log_preds, targets, 1)
    return nll, temps
    
class ComputeNLLs:
    def __init__(self, setup=1, regime="optimal", temps=[], dir=""):
        self.setup = setup
        self.regime = regime
        self.temps = temps
        assert regime == "optimal" or len(temps) > 0
        self.dir = dir
        
    def save(self, model, dataset, setting):
        if model == "VGG16":
            model = "VGG"
        nlls_c_ = {}
        nlls_nc_ = {}
        accs_ = {}
        temps_ = {}
        nlls_c_[(model, dataset, setting)] = self.nlls_c
        nlls_nc_[(model, dataset, setting)] = self.nlls_nc
        accs_[(model, dataset, setting)] = self.accs_global
        temps_[(model, dataset, setting)] = self.temps_global
        result = {"nlls_c": nlls_c_, "nlls_nc":nlls_nc_, "accs":accs_, "temps":temps_}
        with open(self.dir+"/setup%d_%s_%s_%d_%s.pickle"%(self.setup, self.regime, model,\
                                                          dataset, setting), "wb") as fout:
            if self.regime == "optimal":
                pickle.dump(result, fout)
            else: # "grid"
                pickle.dump([self.temps, result], fout)
        
    def get_all_numbers1(self, predictions, targets):
        # predictions are ensemble's probabilities
        acc = np.equal(np.argmax(predictions, axis=1), targets).mean()
        if self.regime == "grid":
            cnlls = []
            for temp in self.temps:
                test_lp = apply_t(np.log(predictions), temp)
                cnll = -get_ll(test_lp, targets)
                cnlls.append(cnll)
            return acc, cnlls, predictions
        else: # "optimal"
            c_nll, temps = metrics_kfold1(np.log(predictions), targets, \
                        n_splits=2, n_runs=5, verbose=False, temp_scale=True)
            nc_nll = -get_ll(np.log(predictions), targets)
            return acc, nc_nll, c_nll, predictions, temps
        
    def get_all_numbers2o(self, list_log_preds, targets):
        # preds is a list of logits
        c_nll, temps = metrics_kfold2(list_log_preds, targets, \
                    n_splits=2, n_runs=5, verbose=False, temp_scale=True)
        preds = np.concatenate([softmax(log_preds)[:, :, None] \
                                for log_preds in list_log_preds], axis=2)
        predictions = np.mean(preds, axis=2)
        nc_nll = -get_ll(np.log(predictions), targets)
        acc = np.equal(np.argmax(predictions, axis=1), targets).mean()
        return acc, nc_nll, c_nll, list_log_preds, temps
        
    def get_ens_quality1_2o(self, preds, targets):
        if self.setup == 1:
            preds = np.concatenate(preds, axis=2)
            predictions = np.mean(preds, axis=2)
            return self.get_all_numbers1(predictions, targets)
        else: # self.setup == 2 and self.regime == "optimal"
            return self.get_all_numbers2o(preds, targets)
    
    def get_ens_quality_cumulative1_2o(self, preds_current, n, new_preds, targets):
        if self.setup == 1:
            reds_current = n/(n+1) * preds_current + 1/(n+1) * new_preds[:, :, 0]
            return self.get_all_numbers1(preds_current, targets)
        else: # self.setup == 2 and self.regime == "optimal"
            preds = preds_current + [new_preds]
            return self.get_all_numbers2o(preds, targets)
    
    def get_all_numbers2g(self, predictions, targets):
        # predictions are ensemble's probabilities
        acc = np.equal(np.argmax(predictions, axis=1), targets).mean()
        cnll = -get_ll(np.log(predictions), targets)
        return acc, cnll, predictions
        
    def get_ens_quality2g(self, preds, targets):
        preds = np.concatenate(preds, axis=2)
        predictions = np.mean(preds, axis=2)
        return self.get_all_numbers2g(predictions, targets)

    def get_ens_quality_cumulative2g(self, preds_current, n, new_preds, targets):
        preds_current = n/(n+1) * preds_current + 1/(n+1) * new_preds[:, :, 0]
        return self.get_all_numbers2g(preds_current, targets)
        
    def compute_nlls(self, logdirs, model_name, num_classes, setting, log,\
                            plen=1, reverse_order=False, max_std=5, max_enslen=10**5):
        loaders, num_classes = data.loaders(
                    "CIFAR%d"%num_classes,
                    "./data/",
                    128,
                    1,
                    "%s_noDA"%("VGG" if model_name == "VGG16" else "ResNet"),
                    True
                )
        targets = np.array(loaders["test"].dataset.targets)

        ll = 1 if not reverse_order else -1
        if not type(logdirs) == list:
            logdirs = [logdirs]
        preds = {}
        for logdir in logdirs:
            for i, p_folder in enumerate(sorted(os.listdir(logdir))):
                if not "ipynb" in p_folder and not "run" in p_folder:
                    p_str = p_folder
                    x = x = p_folder.find("_")
                    if x > 0:
                        p = float(p_folder[plen:x])
                    else:
                        p = float(p_folder[plen:])
                    exp_folders = sorted(os.listdir(logdir+"/"+p_folder))
                    if not p in preds:
                        preds[p] = []
                    for exp_folder in exp_folders:
                        if not "ipynb" in logdir+"/"+p_folder+"/"+exp_folder and\
                        not "run" in logdir+"/"+p_folder+"/"+exp_folder and\
                        not "skipsameseed" in exp_folder:
                            for f in sorted(os.listdir(logdir+"/"+p_folder+"/"+exp_folder))[::ll]:
                                if "predictions" in f:
                                    fn = logdir+"/"+p_folder+"/"+exp_folder+"/"+f
                                    if self.setup == 1:
                                        ppp = softmax(np.float64(np.load(fn)))
                                    else:
                                        ppp = np.float64(np.load(fn))
                                    acc = np.equal(np.argmax(ppp, axis=1), targets).mean()
                                    if acc > 0.15:
                                        preds[p].append(ppp[:, :, None] if self.setup==1\
                                                       else ppp)
        self.nlls_c = {}
        self.nlls_nc = {}
        self.accs_global = {}
        self.temps_global = {}
        ps = sorted(preds.keys())[::-1]
        try:
            for i, p_marker in enumerate(ps):
                if self.setup == 1 or self.regime == "optimal":
                    self.nlls_c[p_marker] = []
                    self.nlls_nc[p_marker] = []
                    self.accs_global[p_marker] = []
                    self.temps_global[p_marker] = []
                    leng = min(len(preds[p_marker]), max_enslen)
                    for l in range(1, leng+1):
                        log.print(p_marker, l)
                        accs, c_nlls, nc_nlls, temps = [], [], [], []
                        if l < leng // 2 + 2:
                            count = min(len(preds[p_marker])//l, max_std)
                            for j in range(count):
                                ret = self.get_ens_quality1_2o(preds[p_marker][j*l:(j+1)*l], targets)
                                if self.regime == "optimal":
                                    acc, nc_nll, c_nll, predictions, temps_ = ret
                                else: # "grid"
                                    acc, c_nll, predictions = ret
                                if acc > 0.15:
                                    accs.append(acc)
                                    c_nlls.append(c_nll)
                                    if self.regime == "optimal":
                                        nc_nlls.append(nc_nll)
                                        temps.append(temps_)
                        else:
                            ret = self.get_ens_quality_cumulative1_2o(predictions, \
                                                       l-1, \
                                                       preds[p_marker][l-1], \
                                                       targets)
                            if self.regime == "optimal":
                                acc, nc_nll, c_nll, predictions, temps_ = ret
                            else: # "grid"
                                acc, c_nll, predictions = ret
                            if acc > 0.15:
                                accs.append(acc)
                                c_nlls.append(c_nll)
                                if self.regime == "optimal":
                                    nc_nlls.append(nc_nll)
                                    temps.append(temps_)
                        self.nlls_c[p_marker].append(c_nlls)
                        self.nlls_nc[p_marker].append(nc_nlls)
                        self.accs_global[p_marker].append(accs)
                        self.temps_global[p_marker].append(temps)
                else: # setup = 2, regime == "grid"
                    self.nlls_c[p_marker] = {}
                    self.nlls_nc[p_marker] = {}
                    self.accs_global[p_marker] = {}
                    self.temps_global[p_marker] = {}
                    for temp in self.temps:
                        log.print(p_marker, temp)
                        preds_p_marker_with_t = [np.exp(apply_t(pr, temp))[:, :, np.newaxis] \
                                                 for pr in preds[p_marker]]
                        self.nlls_c[p_marker][temp] = []
                        self.nlls_nc[p_marker][temp] = []
                        self.accs_global[p_marker][temp] = []
                        self.temps_global[p_marker][temp] = []
                        leng = min(len(preds_p_marker_with_t), max_enslen)
                        for l in range(1, leng+1):
                            accs, c_nlls, nc_nlls, temps = [], [], [], []
                            if l < leng // 2 + 2:
                                count = min(len(preds_p_marker_with_t)//l, max_std)
                                for j in range(count):
                                    ret = self.get_ens_quality2g(preds_p_marker_with_t\
                                                                [j*l:(j+1)*l], targets)
                                    if self.regime == "optimal":
                                        acc, nc_nll, c_nll, predictions, temps_ = ret
                                    else: # "grid"
                                        acc, c_nll, predictions = ret
                                    if acc > 0.15:
                                        accs.append(acc)
                                        c_nlls.append(c_nll)
                                        if self.regime == "optimal":
                                            nc_nlls.append(nc_nll)
                                            temps.append(temps_)
                            else:
                                ret = self.get_ens_quality_cumulative2g(predictions, \
                                                           l-1, \
                                                           preds_p_marker_with_t[l-1], \
                                                           targets)
                                if self.regime == "optimal":
                                    acc, nc_nll, c_nll, predictions, temps_ = ret
                                else: # "grid"
                                    acc, c_nll, predictions = ret
                                if acc > 0.15:
                                    accs.append(acc)
                                    c_nlls.append(c_nll)
                                    if self.regime == "optimal":
                                        nc_nlls.append(nc_nll)
                                        temps.append(temps_)
                            self.nlls_c[p_marker][temp].append(c_nlls)
                            self.nlls_nc[p_marker][temp].append(nc_nlls)
                            self.accs_global[p_marker][temp].append(accs)
                            self.temps_global[p_marker][temp].append(temps)
                self.save(model_name, num_classes, setting)
        except:
            log.print("Except save")
            self.save(model_name, num_classes, setting)
        return self.nlls_c, self.nlls_nc, self.accs_global, self.temps_global

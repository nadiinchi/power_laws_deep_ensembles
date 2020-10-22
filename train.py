import argparse
import os
import sys
import tabulate
import string
import time
import torch
import torch.nn.functional as F
import numpy as np

import data
import models
import utils
import metrics

import logger

def main():


    parser = argparse.ArgumentParser(description='DNN curve training')
    parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                        help='model name (default: None)')
    parser.add_argument('--comment', type=str, default="", metavar='T', help='comment to the experiment')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='save frequency (default: 50)')
    parser.add_argument('--print_freq', type=int, default=1, metavar='N',
                        help='print frequency (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--width', type=int, default=64, metavar='N', help='width of 1 network')
    parser.add_argument('--num-nets', type=int, default=8, metavar='N', help='number of networks in ensemble')
    parser.add_argument('--num-exps', type=int, default=3, metavar='N', help='number of times for executung the whole script')
    parser.add_argument('--not-random-dir', action='store_true',
                        help='randomize dir')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='WD',
                        help='dropout rate for fully-connected layers')
    parser.add_argument('--not-save-weights', action='store_true',
                        help='not save weights')
    parser.add_argument('--lr-shed', type=str, default='standard', metavar='LRSHED',
                        help='lr shedule name (default: standard)')
    parser.add_argument('--shorten_dataset', action='store_true',
                        help='same train set of size N/num_nets for each net')


    args = parser.parse_args()

    letters = string.ascii_lowercase
    
    exp_label = "%s_%s/%s"%(args.dataset, args.model, args.comment)
    if args.num_exps > 1:
        if not args.not_random_dir:
            exp_label += "_%s/"%''.join(random.choice(letters) for i in range(5))
        else:
            exp_label += "/"
    
    np.random.seed(args.seed)
    
    for exp_num in range(args.num_exps):
        args.seed = np.random.randint(1000)
        fmt_list = [('lr', "3.4e"), ('tr_loss', "3.3e"), ('tr_acc', '9.4f'), \
                    ('te_nll', "3.3e"), ('te_acc', '9.4f'), ('ens_acc', '9.4f'),   
                    ('ens_nll', '3.3e'), ('time', ".3f")]
        fmt = dict(fmt_list)
        log = logger.Logger(exp_label, fmt=fmt, base=args.dir)

        log.print(" ".join(sys.argv))
        log.print(args)

        torch.backends.cudnn.benchmark = True
        
        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        loaders, num_classes = data.loaders(
            args.dataset,
            args.data_path,
            args.batch_size,
            args.num_workers,
            args.transform,
            args.use_test
        )
        
        if args.shorten_dataset:
            loaders["train"].dataset.targets = loaders["train"].dataset.targets[:5000]
            loaders["train"].dataset.data = loaders["train"].dataset.data[:5000]

        architecture = getattr(models, args.model)()
        architecture.kwargs["k"] = args.width
        if "VGG" in args.model or "WideResNet" in args.model:
            architecture.kwargs["p"] = args.dropout
 
        if args.lr_shed == "standard":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                alpha = epoch / total_epochs
                if alpha <= 0.5:
                    factor = 1.0
                elif alpha <= 0.9:
                    factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
                else:
                    factor = 0.01
                return factor * base_lr
        elif args.lr_shed == "stair":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                if epoch < total_epochs / 2:
                    factor = 1.0
                else:
                    factor = 0.1
                return factor * base_lr
        elif args.lr_shed == "exp":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                factor = 0.9885 ** epoch
                return factor * base_lr
                


        criterion = F.cross_entropy
        regularizer = None

        ensemble_size = 0
        predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))
        
        for num_model in range(args.num_nets):

            model = architecture.base(num_classes=num_classes, **architecture.kwargs)
            model.cuda()

            optimizer = torch.optim.SGD(
                filter(lambda param: param.requires_grad, model.parameters()),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.wd
            )


            start_epoch = 1
            if args.resume is not None:
                print('Resume training from %s' % args.resume)
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])

            has_bn = utils.check_bn(model)
            test_res = {'loss': None, 'accuracy': None, 'nll': None}
            for epoch in range(start_epoch, args.epochs + 1):
                time_ep = time.time()

                lr = learning_rate_schedule(args.lr, epoch, args.epochs)
                utils.adjust_learning_rate(optimizer, lr)

                train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
                
                ens_acc = None
                ens_nll = None
                if epoch == args.epochs:
                    predictions_logits, targets = utils.predictions(loaders['test'], model)
                    predictions = F.softmax(torch.from_numpy(predictions_logits), dim=1).numpy()
                    predictions_sum = ensemble_size/(ensemble_size+1) \
                                      * predictions_sum+\
                                      predictions/(ensemble_size+1)
                    ensemble_size += 1
                    ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)
                    predictions_sum_log = np.log(predictions_sum+1e-15)
                    ens_nll = -metrics.metrics_kfold(predictions_sum_log, targets, n_splits=2, n_runs=5,\
                                                    verbose=False, temp_scale=True)["ll"]
                    np.save(log.path+'/predictions_run%d' % num_model, predictions_logits)

                if not args.not_save_weights and epoch % args.save_freq == 0:
                    utils.save_checkpoint(
                        log.get_checkpoint(epoch),
                        epoch,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict()
                    )

                time_ep = time.time() - time_ep
                
                if epoch % args.print_freq == 0:
                    test_res = utils.test(loaders['test'], model, \
                                          criterion, regularizer)
                    values = [lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
                              test_res['accuracy'], ens_acc, ens_nll, time_ep]
                    for (k, _), v in zip(fmt_list, values):
                        log.add(epoch, **{k:v})

                    log.iter_info()
                    log.save(silent=True)

            if not args.not_save_weights:
                utils.save_checkpoint(
                    log.path+'/model_run%d.cpt' % num_model,
                    args.epochs,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )
    return log.path    
        
if __name__ == "__main__":
    main()
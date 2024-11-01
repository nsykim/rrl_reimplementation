import os
import logging
import numpy as np
import torch
import torch.distributed as dist
torch.set_num_threads(2)
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from collections import defaultdict
import pandas as pd

from src.dataProcess import *
from src.models import *

DATA_DIR = './dataset'

def get_data_loader(dataset, world_size=1, rank=0, batch_size=32, k=0, pin_memory=False, save_best=True):
    data_path = os.path.join(DATA_DIR, dataset + '.csv')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    # Load data
    X_df, y_df, f_df, label_poss = read_csv(data_path, info_path, shuffle=True)

    # Encode data
    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values

    # K-Fold split
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[k]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Create datasets
    train_set = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
    test_set = TensorDataset(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    # Split train set into train and validation sets
    train_len = int(len(train_set) * 0.95)
    train_sub, valid_set = random_split(train_set, [train_len, len(train_set) - train_len])

    if save_best:  # use validation set for model selections.
        train_set = train_sub

    # Create data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return db_enc, train_loader, valid_loader, test_loader


def train_model(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", init_method='env://', rank=0, world_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    writer = SummaryWriter(args.folder_path)

    dataset = args.data_set
    db_enc, train_loader, valid_loader, test_loader = get_data_loader(dataset,
                                                              batch_size=args.batch_size,
                                                              k=args.ith_kfold,
                                                              save_best=args.save_best)

    # Define model dimensions
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen
    y_fname = db_enc.y_fname
    dim_list = [(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)]
    rrl = RRL(dim_list=dim_list,
              device_id=device,
              log_file=args.log,
              writer=writer,
              save_best=args.save_best,
              estimated_grad=args.estimated_grad,
              use_skip=args.skip,
              save_path=args.model,
              use_nlaf=args.nlaf,
              alpha=args.alpha,
              beta=args.beta,
              gamma=args.gamma,
              temperature=args.temp)

    rrl.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter)
    

def load_model(path, device, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        dim_list=saved_args['dim_list'],
        device_id=device,
        distributed=distributed,
        log_file=log_file,
        estimated_grad=saved_args['estimated_grad'],
        use_skip=saved_args['use_skip'],
        use_nlaf=saved_args['use_nlaf'],
        alpha=saved_args['alpha'],
        beta=saved_args['beta'],
        gamma=saved_args['gamma'])
    stat_dict = checkpoint['model_state_dict']
    for key in list(stat_dict.keys()):
      stat_dict[key[7:]]=stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl

def test_model(args):
    rrl = load_model(args.model, args.device, log_file=args.test_res, distributed=False)
    dataset = args.data_set
    db_enc, train_loader, _, test_loader = get_data_loader(dataset, args.batch_size, args.ith_kfold, save_best=False)
    rrl.test(test_loader=test_loader, set_name='Test')
    with open(args.rrl_file, 'w') as rrl_file:
        rule2weights = rrl.rule_print(db_enc.X_fname, db_enc.y_fname, train_loader, file=rrl_file, mean=db_enc.mean, std=db_enc.std)

    connected_rid = defaultdict(lambda: set())
    ln = len(rrl.net.layer_list) - 1
    for rid, w in rule2weights:
        connected_rid[ln - abs(rid[0])].add(rid[1])
    while ln > 1:
        ln -= 1
        layer = rrl.net.layer_list[ln]
        for r in connected_rid[ln]:
            con_len = len(layer.rule_list[0])
            if r >= con_len:
                opt_id = 1
                r -= con_len
            else:
                opt_id = 0
            rule = layer.rule_list[opt_id][r]
            for rid in rule:
                connected_rid[ln - abs(rid[0])].add(rid[1])


def main(args):
    # dist.destroy_process_group()
    train_model(args)
    test_model(args)
    dist.destroy_process_group()


class Args:
    data_set = 'tictactoe/tic-tac-toe'
    batch_size = 32
    ith_kfold = 0
    save_best = True
    folder_path = './logs'
    log = './logs/log.txt'
    structure = '1@16'
    learning_rate = 0.002
    epoch = 41
    lr_decay_rate = 0.75
    lr_decay_epoch = 200
    weight_decay = 0.0001
    log_iter = 500
    model = './logs/model.pth'
    test_res = './logs/test_results.txt'
    rrl_file = './logs/rules.txt'
    alpha = 0.999
    beta = 8
    gamma = 1
    temp = 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    estimated_grad = True
    skip = True
    nlaf = True # use the novel layer activation function (disjunction & conjunction rather than original...)
args = Args()
main(args)
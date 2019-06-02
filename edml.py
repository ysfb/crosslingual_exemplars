#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:37:52 2018

@author: yusuf
"""
import os
import time
import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EDMLNet(nn.Module):
    def __init__(self, query_dim, document_dim=None, projection_dim=None):
        super(EDMLNet, self).__init__()
        if isinstance(query_dim, str):  # Load config from file
            query_dim, document_dim, projection_dim = loadstr(query_dim)
        self.W1 = nn.Linear(query_dim, projection_dim, bias=False)
        self.W2 = nn.Linear(projection_dim, projection_dim, bias=False)
        width = 1024
        self.width = width
        self.doc1 = nn.Linear(document_dim, width, bias=True)
        self.bn1 = nn.BatchNorm1d(width)
        self.doc2 = nn.Linear(width, width, bias=True)
        self.bn2 = nn.BatchNorm1d(width)
        self.doc3 = nn.Linear(width, width, bias=True)
        self.bn3 = nn.BatchNorm1d(width)
        self.doc4 = nn.Linear(width, projection_dim, bias=True)
        self.bn4 = nn.BatchNorm1d(projection_dim)
        self.bias = nn.Parameter(torch.tensor(0.))
        self.dropout = nn.Dropout(p=0.5)
        self.dimstr = '{},{},{}'.format(query_dim, document_dim,
                                        projection_dim)

    def query_pred(self, x):
        return self.W2(self.W1(x))

    def document_pred(self, x):
        x = F.relu(self.doc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.doc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(self.doc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = F.relu(self.doc4(x))
        x = self.bn4(x)
        x = self.dropout(x)
        return self.W2(x)

    def pred_both(self, inputs):
        query, document = inputs
        return self.query_pred(query), self.document_pred(document)

    def forward(self, inputs):
        q_pred, d_pred = self.pred_both(inputs)
        return torch.sigmoid((q_pred*d_pred).sum(dim=1) + self.bias)

    def save(self, outdir):
        chk_mkdir(outdir)
        with open(os.path.join(outdir, 'nnet.txt'), 'w') as _outfile:
            _outfile.write(self.dimstr)
        torch.save(self.state_dict(), os.path.join(outdir, 'nnet.mdl'))

    @classmethod
    def load_from_dir(cls, nnetdir):
        edml = cls(os.path.join(nnetdir, 'nnet.txt'))
        edml.load_state_dict(torch.load(os.path.join(nnetdir, 'nnet.mdl')))
        return edml


class FeatDataset(Dataset):
    def __init__(self, feat, alignment, query_mat, data_slice,
                 adversary_ratio=2,
                 samps_per_combination=100,
                 transform=None):
        self.samps_per_combination = samps_per_combination
        self.query_mat = query_mat
        self.query_dim = self.query_mat.shape[0]
        self.transform = transform
        self.feat = feat[data_slice]
        self.alignment = alignment[data_slice]
        self.digit_indices = [np.where(self.alignment==i)[0]
                              for i in range(self.query_dim)]
        self.available_classes = sorted(list(set(self.alignment)))
        self.adversary_ratio = adversary_ratio
        self.n_classes = len(self.available_classes)

    def __len__(self):
        return self.samps_per_combination\
                * self.n_classes\
                * (self.n_classes-1)

    def __getitem__(self, index):
        pairs, tr_lab = create_pairs_ind(self.digit_indices,
                                          self.available_classes,
                                          1,
                                          self.adversary_ratio)
        sample = [[self.query_mat[self.alignment[pairs[0]]],
                   self.feat[pairs[1]]],
                  tr_lab]
        if self.transform:
            sample = self.transform(sample)
        return sample


def StopEarly(History, patience=5, mode='min', difference=0.001):
    if mode == 'max':
        h = [-a for a in History]
    else:
        h = History
    History = h
    L = len(History)
    if L <= patience:
        return False
    recent_history = History[L-patience:L]
    antiquity = History[0:L-patience]
    ma = min(recent_history)
    jc = min(antiquity)

    if jc - ma >= difference:
        return False
    else:
        return True


def reduce_lr_on_plateau(History, lr,
                         cooldown=0, patience=5,
                         mode='min', difference=0.001,
                         lr_scale=0.5, lr_min=0.00001,
                         cool_down_patience=None):
    if cool_down_patience and cooldown <= cool_down_patience:
        return lr, cooldown+1
    assert lr_scale < 1
    if mode == 'max':
        h = [-a for a in History]
    else:
        h = History
    History = h
    L = len(History)
    if L <= patience:
        return lr, cooldown+1
    recent_history = History[L-patience:L]
    antiquity = History[0:L-patience]
    ma = min(recent_history)
    jc = min(antiquity)

    if jc - ma >= difference:
        return lr, cooldown+1
    else:
        return max(lr*lr_scale, lr_min), 0


def chk_mkdir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def loadstr(filename, sep=',', dtype=int):
    phrase = open(filename).read().split(sep)
    try:
        diter = iter(dtype)
        assert len(dtype) == len(phrase)
        return [dtype[i](x) for i, x in enumerate(diter)]
    except TypeError:
        return [dtype(x) for x in phrase]


def create_pairs_ind(digit_indices, available_classes,
                     S=1000, adversary_ratio=2):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    n_classes = len(available_classes)
    pairs1 = []
    pairs2 = []
    labels = []

    for i in range(S):
        class_indices = random.sample(range(n_classes), adversary_ratio+1)
        classes = [digit_indices[available_classes[inX]]
                   for inX in class_indices]
        sample_indices = [random.randint(0, len(cls)-1) for cls in classes]
        friend = random.randint(0, len(classes[0]) - 1)
        friend_class = classes[0]
        x1, y1 = friend_class[sample_indices[0]], friend_class[friend]
        pairs1 += [x1]
        pairs2 += [y1]
        labels += [0]
        inX = class_indices[0]
        for i, inY in enumerate(class_indices):
            if not inY == inX:
                foe_class = classes[i]
                x1, y2 = foe_class[sample_indices[i]], friend_class[friend]
                pairs1 += [x1]
                pairs2 += [y2]
                labels += [1]
    pairs1 = np.array(pairs1)
    pairs2 = np.array(pairs2)
    labels = np.array(labels, dtype='float32')
    return [np.array(pairs1), np.array(pairs2)], np.array(labels)


def train_model(model, criterion, optimizer, data_loaders, num_epochs=40,
                verbose=True,
                use_early_stopping=False,
                num_epochs_early_stopping=30,
                delta_early_stopping=1e-4,
                learning_rate_lower_bound=1e-6,
                learning_rate_scale=0.5,
                num_epochs_reduce_lr=12,
                num_epochs_cooldown=8):

    dataset_sizes = {x: len(data_loaders[x].dataset)
                     for x in ['train', 'test']}
    batch_sizes = {x: data_loaders[x].batch_size for x in ['train', 'test']}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 2000
    Losses = {phase: [] for phase in ['train', 'test']}
    History = []
    cooldown = 0
    lr = optimizer.param_groups[0]['lr']
    for epoch in range(num_epochs):
        optimizer.param_groups[0]['lr'] = lr
        tbeg = time.time()
        if verbose:
            print('Epoch {}/{} - lr={}'.format(epoch+1, num_epochs, lr))
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0
            running_correct = 0.
            for batch_no, (feat, lab) in enumerate(data_loaders[phase]):
                b, o = feat
                n = b.size(1)
                feat = [b.view(-1, b.size(-1)).to(device),
                        o.view(-1, o.size(-1)).to(device)]
                lab = lab.view((-1,))
                lab = lab.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(feat)
                    pred_class = torch.sign(torch.clamp(outputs-0.5, 0))
                    loss = criterion(outputs, lab)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * lab.size(0)
                batch_elapse = time.time() - tbeg
                running_correct += torch.sum(pred_class == lab.data)
                eta = int(batch_elapse
                          * (dataset_sizes[phase]
                              // batch_sizes[phase]-batch_no-1)
                          / (batch_no+1))
                if verbose:
                    print('\r\t{} batch: {}/{} batches - ETA: {}s'.format(
                            phase.title(),
                            batch_no+1,
                            dataset_sizes[phase]//batch_sizes[phase]+1,
                            eta
                            ), end='')
            epoch_loss = running_loss / (n * dataset_sizes[phase])
            epoch_acc = running_correct.double() / (n * dataset_sizes[phase])
            Losses[phase].append([float(epoch_loss), float(epoch_acc)])
            if verbose:
                print(' - loss: {:.4f} - acc: {:.4f}'.format(epoch_loss,
                      epoch_acc))
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        History.append(Losses['test'][-1][0])
        rlop = reduce_lr_on_plateau
        lr, cooldown = rlop(History, lr, cooldown, num_epochs_reduce_lr,
                            mode='min', difference=delta_early_stopping,
                            lr_scale=learning_rate_scale,
                            lr_min=learning_rate_lower_bound,
                            cool_down_patience=num_epochs_cooldown)
        if verbose:
            print('\tTime: {}s'.format(int(time.time() - tbeg)))
        if StopEarly(History, patience=num_epochs_early_stopping,
                     mode="min", difference=delta_early_stopping):
            print('Stopping Early.')
            break
    model.load_state_dict(best_model_wts)
    return model, Losses


def main():
    parser = argparse.ArgumentParser(description='EDML in pytorch')
    parser.add_argument('features', help='The training features')
    parser.add_argument('alignment', help='The training alignment')
    parser.add_argument('output_directory', help='Model output directory')

    parser.add_argument('--batch-size', '-b', default=256, type=int,
                        help='Batch size')
    parser.add_argument('--projection-dim', '-w', type=int,
                        help='Dimension of the projection layer')
    parser.add_argument('--samps-per-comb', '-c', default=100, type=int,
                        help='Average number of samples per pair of classes')
    parser.add_argument('--adversary-ratio', '--ar', default=1, type=int,
                        help='Ratio of foes to friends in sampling')
    parser.add_argument('--validation-split', '--spl', type=float, default=0.1,
                        help='Ratio of data to use of validation')

    parser.add_argument('--num-epochs', '-n', default=500,
                        type=int, help='Training number of epochs')
    parser.add_argument('--use-early-stopping', '--es',
                        action='store_false', help='Stop early')
    parser.add_argument('--num-epochs-early-stopping', '--nes',
                        type=int, default=30,
                        help='Number of epochs to wait for '
                        'before early stopping')
    parser.add_argument('--delta-early-stopping', '--des',
                        type=float, default=0.0001,
                        help='The minimum difference for early stopping deference')

    parser.add_argument('--use-model-checkpoint', '--mc',
                        action='store_false',
                        help='Store intermediate models')
    parser.add_argument('--model-checkpoint-period', '--mcp',
                        type=int, default=8,
                        help='Number of epochs after which to '
                        'store intermediate models')

    parser.add_argument('--initial-learning-rate', '--lr',
                        type=float, default=0.001,
                        help='Optimization learning rate')
    parser.add_argument('--learning-rate-lower-bound', '--lr-lower',
                        type=float, default=1e-6,
                        help='Learning rate lower bound')
    parser.add_argument('--learning-rate-scale', '--lr-scale',
                        type=float, default=0.5,
                        help='Learning rate plateau reduction scale')
    parser.add_argument('--num-epochs-reduce-lr', '--nlrr',
                        type=int, default=12,
                        help='Number of epochs to wait before '
                        'reducing the learning rate')
    parser.add_argument('--num-epochs-cooldown', '--nec',
                        type=int, default=8,
                        help='Number of epochs to cooldown before '
                        'tracking plateaus')
    parser.add_argument('--store-pairs', '--sp', action='store_true',
                        help='Store training pairs to disk. '
                        'Useful for reproducing the results')

    args = parser.parse_args()
    feature_file = args.features
    alignment_file = args.alignment
    output_directory = args.output_directory

    batch_size = args.batch_size
    projection_dim = args.projection_dim
    num_epochs = args.num_epochs
    ues = args.use_early_stopping
    use_model_checkpoint = args.use_model_checkpoint
    nes = args.num_epochs_early_stopping
    model_checkpoint_period = args.model_checkpoint_period
    delta_early_stopping = args.delta_early_stopping
    learning_rate = args.initial_learning_rate
    validation_split = args.validation_split
    adversary_ratio = args.adversary_ratio
    samps_per_combination = args.samps_per_comb
    lrb = args.learning_rate_lower_bound
    lrs = args.learning_rate_scale
    nllr = args.num_epochs_reduce_lr
    nec = args.num_epochs_cooldown
    store_pairs = args.store_pairs

    features = np.load(feature_file).astype('float32')
    alignments = np.load(alignment_file).astype('long')
    assert features.shape[0] == alignments.shape[0]
    query_dim = alignments.max() + 1
    query_mat = np.eye(query_dim, dtype='float32')
    document_dim = features.shape[1]

    if not projection_dim:
        projection_dim = document_dim

    validation_length = 0
    train_slice = slice(validation_length, len(alignments))
    val_slice = slice(0, validation_length)

    train_dataset = FeatDataset(features, alignments, query_mat, train_slice,
                                adversary_ratio, samps_per_combination)
    val_dataset = FeatDataset(features, alignments, query_mat, train_slice,
                              adversary_ratio,
                              int(validation_split * samps_per_combination))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=5)  # shuffle=True
    val_dataloader = DataLoader(val_dataset, batch_size=4*batch_size,
                                num_workers=5)
    data_loaders = {'train': train_dataloader,
                    'test': val_dataloader}

    edml = EDMLNet(query_dim, document_dim, projection_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(edml.parameters(), lr=learning_rate)
    edml.to(device)

    edml, Losses = train_model(edml, criterion, optimizer, data_loaders,
                               num_epochs=num_epochs,
                               use_early_stopping=ues,
                               num_epochs_early_stopping=nes,
                               delta_early_stopping=delta_early_stopping,
                               learning_rate_lower_bound=lrb,
                               learning_rate_scale=lrs,
                               num_epochs_reduce_lr=nllr,
                               num_epochs_cooldown=nec)

    chk_mkdir(output_directory)
    edml.save(output_directory)
    torch.save(optimizer.state_dict(), os.path.join(output_directory,
               'optimizer.state'))
    # if store_pairs:
    #     np.save(os.path.join(output_directory, 'pairs0.npy'), pairs[0])
    #     np.save(os.path.join(output_directory, 'pairs1.npy'), pairs[1])
    #     np.save(os.path.join(output_directory, 'labels.npy'), tr_lab)


if __name__ == '__main__':
    main()

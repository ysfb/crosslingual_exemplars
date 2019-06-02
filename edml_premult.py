#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:53:04 2018

@author: yusuf
"""

import os
import argparse
from shutil import copy
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from edml import EDMLNet, chk_mkdir

def save_texts(mats, filenames, fmt='%.6f', dirname=None, sep=' '):
    assert len(mats) == len(filenames)
    for i, mat in enumerate(mats):
        name = filenames[i]
        if dirname:
            name = os.path.join(dirname, name)
        np.savetxt(name, mat, fmt=fmt, delimiter=sep)

def save_mats(mats, dirname, offsets=None, keys=None):
    np.save(os.path.join(dirname, 'data.npy'), mats)
    if offsets:
        copy(offsets,dirname)
    if keys:
        copy(keys, dirname)
def main():
    parser = argparse.ArgumentParser(description='Premultiply the features with '
                                     'the EDML matrices for search')
    parser.add_argument('nnet_dir', help='The directory in which the EDML network is stored')
    parser.add_argument('document_dir', help='A directory containing the document. Expected to have three files:- '
                        'data.npy: A numpy array of the entire document'
                        'offsets_data.npy: An array of utterance offsets'
                        'keys_data.txt: A list of utterance names')
    parser.add_argument('output_dir', help='The directory into which to store')
    parser.add_argument('--batch-size', '-b', type=int, default=1024,
                        help='Number of elements in each processed batch')
    parser.add_argument('--clusters-file', '-c',
                        help='File that maps state names to query dimensions')
    parser.add_argument('--save-output-as-npy', action='store_true',
                        help='Store the output as numpy files instead of text')
    
    args = parser.parse_args()
    nnet_dir = args.nnet_dir
    document_dir = args.document_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    clusters_file = args.clusters_file
    save_output_as_npy = args.save_output_as_npy
    
    feats = os.path.join(document_dir, 'data.npy')
    keys_file = os.path.join(document_dir, 'keys_data.txt')
    offsets = os.path.join(document_dir,'offsets_data.npy')
    
    edml = EDMLNet.load_from_dir(nnet_dir)
    edml.eval()
    edml.to(device)
    x = np.load(feats)
    offs = np.load(offsets)
    keys = [fl.strip() for fl in open(keys_file).readlines()]
    
    x_tensor_batches = torch.split(torch.tensor(x), batch_size)
    y_batches = [edml.document_pred(batch.to(device)).data.to('cpu').numpy() for batch in x_tensor_batches]
    y = np.concatenate(y_batches)
    y_split = np.split(y, offs)
    
    query_dim = edml.W1.weight.size()[1]
    query_mat = torch.eye(query_dim).to(device)
    z = edml.query_pred(query_mat).data.to('cpu').numpy()
    
    bias = edml.bias.item()
    
    odir1 = os.path.join(output_dir, 'feat')
    odir2 = os.path.join(output_dir, 'phones')
    chk_mkdir(odir1)
    chk_mkdir(odir2)
    if len(y_split) != len(keys):
        print ("\t\t\tError: The number of feature files does not match the number of keys; {}!={}. Skipping".format(len(y_split), len(keys)))
        exit(1)
        
    if save_output_as_npy:
        save_mats(y, odir1, offsets, keys_file)
    else:
        save_texts(y_split, keys, dirname=odir1)
    
    with open(os.path.join(odir2, 'bias.txt'), 'w') as _bias:
        _bias.write(str(bias))
    if clusters_file:
        clusters = {line.split(':')[0]: line.split(':')[1] for line in open(clusters_file).readlines()}
    else:
        clusters = {str(x): str(x) for x in range(query_dim)}
    save_texts([_[:, np.newaxis].T for _ in z], [clusters[str(i)] for i in range(query_dim)], dirname=odir2)
    
if __name__=='__main__':
    main()

"""Data Layer Classes"""
from os import listdir, path
from random import shuffle
import numpy as np
import torch

class PlaySongRecDataProvider:
    
    def __init__(self, raw_data, params, test=False):
        self.data = raw_data
        self._vector_dim = params['vector_dim']
        self._batch_size = params['batch_size']
    
    def iterate_one_epoch(self):
        data = self.data
        shuffle(data)
        s_ind = 0
        e_ind = self._batch_size
        while e_ind <= len(data):
            if (len(data) - e_ind) < self._batch_size:
                mini_batch = torch.FloatTensor(data[s_ind:len(data)])
            else:
                mini_batch = torch.FloatTensor(data[s_ind:e_ind])
            s_ind += self._batch_size
            e_ind += self._batch_size
            yield  mini_batch

    
    def iterate_test_epoch(self):
        data = self.data
        s_ind = 0
        e_ind = min(self._batch_size,len(data))
        while e_ind <= len(data):
            if (len(data) - e_ind) < self._batch_size:
                mini_batch = torch.FloatTensor(data[s_ind:len(data)])
            else:
                mini_batch = torch.FloatTensor(data[s_ind:e_ind])
            s_ind += self._batch_size
            e_ind += self._batch_size
            yield  mini_batch
    
    
    
    @property
    def vector_dim(self):
        return self._vector_dim

    @property
    def userIdMap(self):
        return self._user_id_map

    @property
    def itemIdMap(self):
        return self._item_id_map

    @property
    def params(self):
        return self._params
    
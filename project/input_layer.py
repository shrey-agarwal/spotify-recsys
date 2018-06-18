"""Data Layer Classes"""
from os import listdir, path
from random import shuffle
import numpy as np
import torch

class PlaySongRecDataProvider:
    
    def __init__(self, raw_data, params, test=False):
        data = {}
        if test == True:
            
            for key,songs in enumerate(raw_data):
                data[key] = [ (int(song),len(songs)-i) for i,song in enumerate(songs)]

        else:
            for key,songs in enumerate(raw_data):
                data[key] = [ (int(song),1) for song in songs]
        
        self.data = data
        songs = list(data.values())
        songs = [s[0] for song in songs for s in song]
        songs = set(songs)
        self._vector_dim = params['vector_dim']
        self._batch_size = params['batch_size']
    
    def iterate_one_epoch(self):
        data = self.data
        keys = list(data.keys())
        shuffle(keys)
        s_ind = 0
        e_ind = self._batch_size
        
        while e_ind <= len(keys):
            local_ind = 0
            inds1 = []
            inds2 = []
            vals = []
            for ind in range(s_ind, e_ind):
                inds2 += [v[0] for v in data[keys[ind]]]
                inds1 += [local_ind]*len([v[0] for v in data[keys[ind]]])
                vals += [v[1] for v in data[keys[ind]]]
                local_ind += 1
            
            #print(inds2)
            i_torch = torch.LongTensor([inds1, inds2])
            v_torch = torch.FloatTensor(vals)
            
            mini_batch = torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([self._batch_size, self._vector_dim]))
            s_ind += self._batch_size
            e_ind += self._batch_size
            yield  mini_batch

            
    def iterate_one_epoch_test(self, keys):
        data = self.data
        shuffle(keys)
        s_ind = 0
        e_ind = self._batch_size
        
        while e_ind <= len(keys):
            local_ind = 0
            inds1 = []
            inds2 = []
            vals = []
            src_inds1 = []
            src_inds2 = []
            src_vals = []
            for i,ind in enumerate(keys[s_ind:e_ind]):
                inds2 += [v[0] for v in data[i]]
                inds1 += [local_ind]*len([v[0] for v in data[i]])
                vals += [v[1] for v in data[i]]
                
                src_inds2 = [v[0] for v in self.src_data[ind]]
                src_inds1 = [local_ind] * len([v[0] for v in self.src_data[ind]])
                src_vals = [v[1] for v in self.src_data[ind]]
                local_ind += 1
            
            i_torch = torch.LongTensor([inds1, inds2])
            v_torch = torch.FloatTensor(vals)
            src_i_torch = torch.LongTensor([src_inds1, src_inds2])
            src_v_torch = torch.FloatTensor(src_vals)
            mini_batch = (torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([self._batch_size, self._vector_dim])),
                        torch.sparse.FloatTensor(src_i_torch, src_v_torch, torch.Size([self._batch_size, self._vector_dim])))
            
            s_ind += self._batch_size
            e_ind += self._batch_size
            yield  mini_batch
            
            
            
    def iterate_one_epoch_eval(self, for_inf=False):
        keys = list(self.data.keys())
        s_ind = 0
        while s_ind < len(keys):
            inds1 = [0] * len([v[0] for v in self.data[keys[s_ind]]])
            inds2 = [v[0] for v in self.data[keys[s_ind]]]
            vals = [v[1] for v in self.data[keys[s_ind]]]

            src_inds1 = [0] * len([v[0] for v in self.src_data[keys[s_ind]]])
            src_inds2 = [v[0] for v in self.src_data[keys[s_ind]]]
            src_vals = [v[1] for v in self.src_data[keys[s_ind]]]

            i_torch = torch.LongTensor([inds1, inds2])
            v_torch = torch.FloatTensor(vals)

            src_i_torch = torch.LongTensor([src_inds1, src_inds2])
            src_v_torch = torch.FloatTensor(src_vals)

            mini_batch = (torch.sparse.FloatTensor(i_torch, v_torch, torch.Size([1, self._vector_dim])),
                        torch.sparse.FloatTensor(src_i_torch, src_v_torch, torch.Size([1, self._vector_dim])))
            s_ind += 1
        if not for_inf:
            yield  mini_batch
        else:
            yield mini_batch, keys[s_ind - 1]

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
    
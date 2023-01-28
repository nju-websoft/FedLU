from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self,triples,nentity,nrelation,negative_sample_size,mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head,self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        positive_sample = self.triples[idx]
        head,relation,tail = positive_sample
        
        subsampling_weight = self.count[(head,relation)] + self.count[(tail,-relation-1)]
        subsampling_weight = torch.sqrt(1/torch.Tensor([subsampling_weight]))

        negative_sample_list = list()
        negative_sample_size = 0
        while(negative_sample_size<self.negative_sample_size):
            negative_sample = np.random.randint(self.nentity,size=self.negative_sample_size*2)
            if(self.mode=="head-batch"):
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation,tail)],
                    assume_unique=True,
                    invert=True
                )
            elif(self.mode=="tail-batch"):
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head,relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError("training batch mode %s not supported" %self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample,negative_sample,subsampling_weight,self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data],dim=0)
        negative_sample = torch.stack([_[1] for _ in data],dim=0)
        subsample_weight = torch.cat([_[2] for _ in data],dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples,start=4):
        count = dict()
        for head,relation,tail in triples:
            if((head,relation) not in count):
                count[(head,relation)] = start
            else:
                count[(head,relation)] += 1
            
            if((tail,-relation-1) not in count):
                count[(tail,-relation-1)] = start
            else:
                count[(tail,-relation-1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head = dict()
        true_tail = dict()
        
        for head,relation,tail in triples:
            if((head,relation) not in true_tail):
                true_tail[(head,relation)] = list()
            true_tail[(head,relation)].append(tail)
            if((relation,tail) not in true_head):
                true_head[(relation,tail)] = list()
            true_head[(relation,tail)].append(head)
        
        for head,relation in true_tail:
            true_tail[(head,relation)] = np.array(list(set(true_tail[(head,relation)])))
        for relation,tail in true_head:
            true_head[(relation,tail)] = np.array(list(set(true_head[(relation,tail)])))
        
        return true_head,true_tail

class TestDataset(Dataset):
    def __init__(self,triples,all_true_triples,nentity,nrelation,mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        head,relation,tail = self.triples[idx]
        if(self.mode=="head-batch"):
            tmp = [(0,rand_head) if(rand_head,relation,tail) not in self.triple_set
                    else (-1,head) for rand_head in range(self.nentity)]
            tmp[head] = (0,head)
        elif(self.mode=="tail-batch"):
            tmp = [(0,rand_tail) if (head,relation,rand_tail) not in self.triple_set
                    else (-1,tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0,tail)
        else:
            raise ValueError("negative batch mode %s not supported" %self.mode)
        
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:,0].float()
        negative_sample = tmp[:,1]
        positive_sample = torch.LongTensor((head,relation,tail))

        return positive_sample,negative_sample,filter_bias,self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample,negative_sample,filter_bias,mode

class TestDataset_Partial(Dataset):
    def __init__(self,triples,all_true_triples,client_ent,nentity,nrelation,mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.client_ent = set(client_ent)
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        head,relation,tail = self.triples[idx]
        if self.mode=="head-batch":
            tmp = [(-1,head) if ((rand_head not in self.client_ent) or ((rand_head,relation,tail) in self.triple_set))
                    else (0,rand_head) for rand_head in range(self.nentity)]
            tmp[head] = (0,head)
        elif self.mode=="tail-batch":
            tmp = [(-1,tail) if((rand_tail not in self.client_ent) or ((head,relation,rand_tail) in self.triple_set))
                    else (0,rand_tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0,tail)
        else:
            raise ValueError("negative batch mode %s not supported" %self.mode)
        
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:,0].float()
        negative_sample = tmp[:,1]
        positive_sample = torch.LongTensor((head,relation,tail))

        return positive_sample,negative_sample,filter_bias,self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample,negative_sample,filter_bias,mode
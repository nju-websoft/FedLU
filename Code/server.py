from functools import total_ordering
import numbers
import os
from copy import deepcopy
import random
import string
import itertools
from turtle import left
import numpy as np
from numpy.lib.function_base import copy
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F

class Server:
    def __init__(self,args):
        self.args = deepcopy(args)
        self.nentity = self.load_entities()
    
    def load_entities(self):
        args = self.args
        local_file_dir = args.local_file_dir
        client_entities_dict = dict()
        for client_dir in os.listdir(local_file_dir):
            client_seq = int(client_dir)
            client_entities = []
            with open(os.path.join(local_file_dir,client_dir,"entities.dict"),"r",encoding="utf-8")as fin:
                for line in fin.readlines():
                    _,label = line.strip().split()
                    client_entities.append(label)
            client_entities_dict[client_seq] = client_entities
        self.client_entities_dict = client_entities_dict
        all_entities = []
        for client_seq in client_entities_dict.keys():
            all_entities.extend(client_entities_dict[client_seq])
        all_entities = list(set(all_entities))
        nentity = len(all_entities)
        client_entities_mapping = dict()
        for client_seq in client_entities_dict.keys():
            client_entities = client_entities_dict[client_seq]
            client_entities_mapping[client_seq] = [all_entities.index(client_entity) for client_entity in client_entities]
        self.client_entities_mapping = client_entities_mapping
        return nentity

    def generate_global_embedding(self):
        args = self.args
        hidden_dim = args.hidden_dim*2 if args.double_entity_embedding else args.hidden_dim
        embedding_range = torch.Tensor([(args.gamma+args.epsilon)/hidden_dim])
        self.global_entity_embedding = torch.zeros(self.nentity,hidden_dim)
        nn.init.uniform(tensor=self.global_entity_embedding,a=-embedding_range.item(),b=embedding_range.item())
    
    def assign_embedding(self):
        entity_embedding_dict = dict()
        for client_seq in self.client_entities_mapping.keys():
            client_embedding = self.global_entity_embedding[self.client_entities_mapping[client_seq]]
            entity_embedding_dict[client_seq] = client_embedding
        return entity_embedding_dict
    
    def aggregate_embedding(self,entity_embedding_dict):
        args = self.args
        hidden_dim = args.hidden_dim*2 if args.double_entity_embedding else args.hidden_dim
        later_global_embedding = torch.zeros(self.nentity,hidden_dim)
        weight = torch.zeros(self.nentity)
        for client_seq in entity_embedding_dict.keys():
            if args.agg=="distance":
                client_distance_norm = (self.global_entity_embedding[self.client_entities_mapping[client_seq]]-entity_embedding_dict[client_seq].cpu().detach()).norm(p=2,dim=1)
                client_distance_norm = torch.exp(client_distance_norm)
                weight[self.client_entities_mapping[client_seq]] += client_distance_norm
                later_global_embedding[self.client_entities_mapping[client_seq]] += client_distance_norm.view(client_distance_norm.shape[0],1)*entity_embedding_dict[client_seq].cpu().detach()
            elif args.agg=="similarity":
                client_similarity_score = torch.cosine_similarity(self.global_entity_embedding[self.client_entities_mapping[client_seq]],entity_embedding_dict[client_seq].cpu().detach(),dim=1)
                client_similarity_score = torch.exp(client_similarity_score)
                weight[self.client_entities_mapping[client_seq]] += client_similarity_score
                later_global_embedding[self.client_entities_mapping[client_seq]] += client_similarity_score.view(client_similarity_score.shape[0],1)*entity_embedding_dict[client_seq].cpu().detach()
            elif args.agg=="weighted":
                weight[self.client_entities_mapping[client_seq]] += 1
                later_global_embedding[self.client_entities_mapping[client_seq]] += entity_embedding_dict[client_seq].cpu().detach()
            else:
                raise ValueError("Aggregation method should be chosen among weighted/distance/similarity")
        standard = torch.ones(weight.shape)
        weight = torch.where(weight>0,weight,standard)
        weight = weight.view(weight.shape[0],1)
        later_global_embedding/=weight
        self.global_entity_embedding = later_global_embedding
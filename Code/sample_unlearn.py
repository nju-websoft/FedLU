import os
import math
import torch
import random
import numpy as np

seed = 1998111213
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def read_triples(file_path):
    data = []
    with open(file_path,"r",encoding="utf-8") as fin:
        for line in fin.readlines():
            h,r,t = line.strip().split()
            data.append((h,r,t))
    return data

def write_triples(file_path,data):
    with open(file_path,"w",encoding="utf-8") as fout:
        for (h,r,t) in data:
            fout.write(h+"\t"+r+"\t"+t+"\n")

def random_select(data,frac=0.01):
    entity_count = dict()
    relation_count = dict()
    for (h,r,t) in data:
        if h not in entity_count:
            entity_count[h] = 0
        if r not in relation_count:
            relation_count[r] = 0
        if t not in entity_count:
            entity_count[t] = 0
        entity_count[h] += 1
        relation_count[r] += 1
        entity_count[t] += 1
    result_size = math.ceil(len(data)*frac)
    result = set()
    batch_size = math.ceil(result_size/10)
    while(len(result)<result_size):
        result_batch = random.sample(data,batch_size)
        for (h,r,t) in result_batch:
            if(entity_count[h]<=1 or relation_count[r]<=1 or entity_count[t]<=1):
                continue
            else:
                entity_count[h] -= 1
                relation_count[r] -= 1
                entity_count[t] -= 1
                result.add((h,r,t))
    unlearn = list(result)[:result_size]
    false = []
    for (h,r,t) in unlearn:
        t_f = random.sample(entity_count.keys(),1)[0]
        while(t_f==h or t_f==t or ((h,r,t_f) in data)):
            t_f = random.sample(entity_count.keys(),1)[0]
        false.append((h,r,t_f))
    return unlearn,false

def relation_centric_select(data,frac=0.01):
    entity_count = dict()
    relation_count = dict()
    relation2triple = dict()
    for (h,r,t) in data:
        if h not in entity_count:
            entity_count[h] = 0
        if r not in relation_count:
            relation_count[r] = 0
        if t not in entity_count:
            entity_count[t] = 0
        entity_count[h] += 1
        relation_count[r] += 1
        entity_count[t] += 1
        if r not in relation2triple:
            relation2triple[r] = []
        relation2triple[r].append((h,r,t))
    result_size = math.ceil(len(data)*frac)
    result = set()
    relation_selected = set()
    while(len(result)<result_size):
        relation_batch_size = math.ceil(frac*len(relation2triple.keys()))
        relation_batch = random.sample(relation2triple.keys(),relation_batch_size)
        for r in relation_batch:
            if r not in relation_selected:
                relation_selected.add(r)
                triple_batch_size = math.ceil(len(relation2triple[r])/10)
                triple_batch = random.sample(relation2triple[r],triple_batch_size)
                for (h,r,t) in triple_batch:
                    if(entity_count[h]<=1 or entity_count[t]<=1 or relation_count[r]<=1):
                        continue
                    else:
                        entity_count[h] -= 1
                        entity_count[t] -= 1
                        relation_count[r] -= 1
                        result.add((h,r,t))
                if len(result)>=result_size:
                    break
    print("number of selected identical relations %i"%(len(relation_selected)))
    unlearn = list(result)[:result_size]
    false = []
    for (h,r,t) in unlearn:
        t_f = random.sample(entity_count.keys(),1)[0]
        while(t_f==h or t_f==t or ((h,r,t_f) in data)):
            t_f = random.sample(entity_count.keys(),1)[0]
        false.append((h,r,t_f))
    return unlearn,false

def construct_unlearn(inpath,frac=0.01,type="random"):
    trainfile = os.path.join(inpath,"train.txt")
    unlearnfile = os.path.join(inpath,"unlearn.txt")
    falsefile = os.path.join(inpath,"false.txt")
    train = read_triples(trainfile)
    print("total number: %i"%(len(train)))
    if type=="random":
        unlearn,false = random_select(train,frac=frac)
    elif type=="relation":
        unlearn,false  = relation_centric_select(train,frac=frac)
    else:
        raise ValueError("type must be random/relation, default random")
    print("unlearn number: %i"%(len(unlearn)))
    print("false number: %i"%(len(false)))
    write_triples(unlearnfile,unlearn)

inpath="../Data/FB15k-237/C3FL/0"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C3FL/1"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C3FL/2"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C5FL/0"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C5FL/1"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C5FL/2"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C5FL/3"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C5FL/4"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/0"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/1"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/2"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/3"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/4"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/5"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/6"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/7"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/8"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/C10FL/9"
construct_unlearn(inpath,type="random")

inpath="../Data/FB15k-237/R3FL/0"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R3FL/1"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R3FL/2"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R5FL/0"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R5FL/1"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R5FL/2"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R5FL/3"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R5FL/4"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/0"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/1"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/2"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/3"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/4"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/5"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/6"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/7"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/8"
construct_unlearn(inpath,type="random")
inpath="../Data/FB15k-237/R10FL/9"
construct_unlearn(inpath,type="random")
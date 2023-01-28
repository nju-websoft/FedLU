import os
from math import ceil
from random import sample
from copy import deepcopy
def construct_fldataset(in_dir,out_dir,valid_frac=0.1,test_frac=0.1):
    filelist = os.listdir(in_dir)
    global_eset = set()
    global_rset = set()
    for filenm in filelist:
        if ".txt" in filenm:
            seq = filenm[:-4]
            if not os.path.exists(os.path.join(out_dir,seq)):
                os.makedirs(os.path.join(out_dir,seq))
            eset = set()
            rset = set()
            entity_count = dict()
            relation_count = dict()
            triples = []
            with open(os.path.join(in_dir,filenm),"r",encoding="utf-8")as fin:
                for line in fin.readlines():
                    h,r,t = line.strip().split()
                    eset.add(h)
                    eset.add(t)
                    rset.add(r)
                    if h not in entity_count:
                        entity_count[h] = 0
                    if r not in relation_count:
                        relation_count[r] = 0
                    if t not in entity_count:
                        entity_count[t] = 0
                    entity_count[h] += 1
                    entity_count[t] += 1
                    relation_count[r] += 1
                    triples.append((h,r,t))
            train_triples = deepcopy(triples)
            global_eset |= eset
            global_rset |= rset
            with open(os.path.join(out_dir,seq,"entities.dict"),"w",encoding="utf-8")as fout:
                i=0
                for e in eset:
                    fout.write("%s\t%s\n"%(i,e))
                    i += 1
            with open(os.path.join(out_dir,seq,"relations.dict"),"w",encoding="utf-8")as fout:
                i=0
                for r in rset:
                    fout.write("%s\t%s\n"%(i,r))
                    i += 1
            if valid_frac>0.0:
                valid_size = ceil(len(triples)*valid_frac)
                valid_triples = set()
                valid_batchsize = ceil(valid_size/10)
                while(len(valid_triples)<valid_size):
                    valid_batch = sample(triples,valid_batchsize)
                    for (h,r,t) in valid_batch:
                        if(entity_count[h]<=1 or relation_count[r]<=1 or entity_count[t]<=1):
                            continue
                        else:
                            entity_count[h] -= 1
                            relation_count[r] -= 1
                            entity_count[t] -= 1
                            valid_triples.add((h,r,t))
                valid_triples = list(valid_triples)[:valid_size]
                with open(os.path.join(out_dir,seq,"valid.txt"),"w",encoding="utf-8")as fout:
                    for (h,r,t) in valid_triples:
                        fout.write("%s\t%s\t%s\n"%(h,r,t))
                train_triples = list(set(train_triples)-set(valid_triples))
            if test_frac>0.0:
                test_size = ceil(len(triples)*test_frac)
                test_triples = set()
                test_batchsize = ceil(test_size/10)
                triples = list(set(triples)-set(valid_triples))
                while(len(test_triples)<test_size):
                    test_batch = sample(triples,test_batchsize)
                    for (h,r,t) in test_batch:
                        if (entity_count[h]<=1 or relation_count[r]<=1 or entity_count[t]<=1):
                            continue
                        else:
                            entity_count[h] -= 1
                            relation_count[r] -= 1
                            entity_count[t] -= 1
                            test_triples.add((h,r,t))
                test_triples = list(test_triples)[:test_size]
                with open(os.path.join(out_dir,seq,"test.txt"),"w",encoding="utf-8")as fout:
                    for (h,r,t) in test_triples:
                        fout.write("%s\t%s\t%s\n"%(h,r,t))
                train_triples = list(set(train_triples)-set(test_triples))
            with open(os.path.join(out_dir,seq,"train.txt"),"w",encoding="utf-8")as fout:
                    for (h,r,t) in train_triples:
                        fout.write("%s\t%s\t%s\n"%(h,r,t))
        else:
            continue

in_dir = "../Data/FB15k-237/R3"
out_dir = "../Data/FB15k-237/R3FL"
construct_fldataset(in_dir,out_dir)
in_dir = "../Data/FB15k-237/C3"
out_dir = "../Data/FB15k-237/C3FL"
construct_fldataset(in_dir,out_dir)
in_dir = "../Data/FB15k-237/R5"
out_dir = "../Data/FB15k-237/R5FL"
construct_fldataset(in_dir,out_dir)
in_dir = "../Data/FB15k-237/C5"
out_dir = "../Data/FB15k-237/C5FL"
construct_fldataset(in_dir,out_dir)
in_dir = "../Data/FB15k-237/R10"
out_dir = "../Data/FB15k-237/R10FL"
construct_fldataset(in_dir,out_dir)
in_dir = "../Data/FB15k-237/C10"
out_dir = "../Data/FB15k-237/C10FL"
construct_fldataset(in_dir,out_dir)
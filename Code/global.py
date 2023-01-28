from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
from pyexpat import model
import random
from statistics import mode

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import KGEModel
from dataloader import TrainDataset,TestDataset,TestDataset_Partial
from itertools import chain

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="train, valid, test and unlearn kge models collaboratively",
        usage = "controller.py [<args>] [-h | --help]"
    )
    parser.add_argument("--client_num",type=int,default=3,help="client num, type int, default 3")
    parser.add_argument("--local_file_dir",type=str,default="../Data/FB15k-237/C3Global",help="local file directory, type str, default ../Data/FB15k-237/C3Global")
    parser.add_argument("--save_dir",type=str,default="../Output/FB15k-237/C3FL",help="save dir, type str, default ../Output/FB15k-237/C3Global")
    parser.add_argument("--cuda",action="store_true",help="use GPU, store true")
    parser.add_argument("--model",type=str,default="TransE",help="model, type str, default TransE")
    parser.add_argument("--double_entity_embedding",action="store_true",help="double entity embedding, store true")
    parser.add_argument("--double_relation_embedding",action="store_true",help="double relation embedding, store true")
    parser.add_argument("--max_epoch",type=int,default=300,help="max epoch, type int, default 300")
    parser.add_argument("--valid_epoch",type=int,default=10,help="valid epoch, type int, default 10")
    parser.add_argument("--early_stop_epoch",type=int,default=3,help="early stop epoch, type int, default 3")
    parser.add_argument("--cpu_num",type=int,default=16,help="cpu num, type int, default 16")
    parser.add_argument("--negative_sample_size",type=int,default=256,help="negative sample size, type int, default 256")
    parser.add_argument("--negative_adversarial_sampling",action="store_true",help="negative adversarial sampling, store true")
    parser.add_argument("--adversarial_temperature",type=float,default=1.0,help="float, adversarial temperature, default 1.0")
    parser.add_argument("--uni_weight",action="store_true",help="uni weight, store true")
    parser.add_argument("--regularization",type=float,default=0.0,help="regularization, type float, default 0.0")
    parser.add_argument("--batch_size",type=int,default=512,help="batch size, type int, default 1024")
    parser.add_argument("--hidden_dim",type=int,default=256,help="hidden dim, type int, default 512")
    parser.add_argument("--learning_rate",type=float,default=1e-4,help="learning rate, type float, default 1e-4")
    parser.add_argument("--gamma",type=float,default=9.0,help="gamma, type float, default 9.0")
    parser.add_argument("--epsilon",type=float,default=2.0,help="epsilon, type float, default 2.0")
    parser.add_argument("--test_batch_size",type=int,default=32,help="test batch size, type int, default 32")
    parser.add_argument("--log_epoch",type=int,default=1,help="log epoch, type int, default 1")
    parser.add_argument("--test_log_step",type=int,default=200,help="test log step, type int, default 200")
    parser.add_argument("--fed_mode",type=str,default="FedAvg",help="fed mode, type str, default FedAvg")
    parser.add_argument("--mu",type=float,default=0.0,help="mu, type float, default 0.0")
    parser.add_argument("--mu_decay",action="store_true",help="mu decay, store true")
    args = parser.parse_args(args)
    if args.local_file_dir is None:
        raise ValueError("local file dir must be set")
    if args.model=="RotatE":
            args.double_entity_embedding=True
            args.negative_adversarial_sampling=True
    elif args.model=="ComplEx":
            args.double_entity_embedding=True
            args.double_relation_embedding=True
    return args
    
def read_triple(file_path,entity2id,relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h,r,t = line.strip().split("\t")
            triples.append((entity2id[h],relation2id[r],entity2id[t]))
    return triples

def save_model(model,optimizer,save_varible_list,args,save_path):
    argparse_dict = vars(args)
    with open(os.path.join(save_path,"config.json"),"w") as fjson:
        json.dump(argparse_dict,fjson)
    torch.save(
        {
            **save_varible_list,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict()
        },
        os.path.join(save_path,"checkpoint")
    )
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(os.path.join(save_path,"entity_embedding"),entity_embedding)
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(os.path.join(save_path,"relation_embedding"),relation_embedding)

def log_metrics(mode,epoch,metrics):
    print("Mode: %s"%(mode))
    print("Epoch: %i"%(epoch))
    for metric in metrics.keys():
        print("%s: %f"%(metric,metrics[metric]))

def main(args):
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.local_file_dir,"entities.dict"),"r") as fin:
        entity2id = dict()
        for line in fin:
            id,entity = line.strip().split("\t")
            entity2id[entity] = int(id)
    with open(os.path.join(args.local_file_dir,"relations.dict"),"r") as fin:
        relation2id = dict()
        for line in fin:
            id,relation = line.strip().split("\t")
            relation2id[relation] = int(id)
    client_es = []
    for i in range(args.client_num):
        client_e = []
        with open(os.path.join(args.local_file_dir,"e"+str(i)+".set"),"r",encoding="utf-8") as fin:
            for line in fin.readlines():
                client_e.append(line.strip())
        client_es.append([entity2id[entity] for entity in client_e])
    nentity = len(entity2id)
    nrelation = len(relation2id)
    args.nentity = nentity
    args.nrelation = nrelation
    train_triples = read_triple(os.path.join(args.local_file_dir,"train.txt"),entity2id,relation2id)
    print("#train: %i" %(len(train_triples)))
    valid_triples = read_triple(os.path.join(args.local_file_dir,"valid.txt"),entity2id,relation2id)
    print("#valid: %i" %(len(valid_triples)))
    test_sets = []
    test_triples = []
    for i in range(args.client_num):
        test_sets.append(read_triple(os.path.join(args.local_file_dir,"test"+str(i)+".txt"),entity2id,relation2id))
        test_triples.extend(test_sets[i])
    print("#test: %i" %(len(test_triples)))

    all_true_triples = train_triples+valid_triples+test_triples

    kge_model = KGEModel(
        model_name = args.model,
        nentity  = nentity,
        nrelation  = nrelation,
        hidden_dim = args.hidden_dim,
        gamma = args.gamma,
        double_entity_embedding = args.double_entity_embedding,
        double_relation_embedding = args.double_relation_embedding
    )
    for name,param in kge_model.named_parameters():
        print("parameter %s: %s, require_grad=%s" %(name,str(param.size()),str(param.requires_grad)))
    if args.cuda:
        kge_model = kge_model.cuda()
    
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples,nentity,nrelation,args.negative_sample_size,"head-batch"),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = max(0,args.cpu_num//2),
        collate_fn = TrainDataset.collate_fn
    )
    train_dataloader_tail = DataLoader(
        TrainDataset(train_triples,nentity,nrelation,args.negative_sample_size,"tail-batch"),
        batch_size = args.batch_size,
        shuffle = True,
        num_workers=max(0,args.cpu_num//2),
        collate_fn = TrainDataset.collate_fn
    )
    train_iterator = list(chain.from_iterable(zip(train_dataloader_head,train_dataloader_tail)))
    valid_dataloader_head = DataLoader(
        TestDataset(valid_triples,all_true_triples,nentity,nrelation,"head-batch",),
        batch_size=args.test_batch_size,
        num_workers=max(0,args.cpu_num//2),
        collate_fn=TestDataset.collate_fn,
    )
    valid_dataloader_tail = DataLoader(
        TestDataset(valid_triples,all_true_triples,nentity,nrelation,"tail-batch",),
        batch_size=args.test_batch_size,
        num_workers=max(0,args.cpu_num//2),
        collate_fn=TestDataset.collate_fn,
    )
    valid_dataset_list = [valid_dataloader_head, valid_dataloader_tail]
    test_dataset_lists = []
    for i in range(args.client_num):
        test_dataloader_head = DataLoader(
            TestDataset_Partial(test_sets[i],all_true_triples,client_es[i],nentity,nrelation,"head-batch",),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
        )
        test_dataloader_tail = DataLoader(
            TestDataset_Partial(test_sets[i],all_true_triples,client_es[i],nentity,nrelation,"tail-batch",),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
        )
        test_dataset_lists.append([test_dataloader_head,test_dataloader_tail])

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()),
        lr = args.learning_rate
    )
    training_logs = []
    bad_epoch = 0
    best_mrr = 0
    epoch = 0
    for epoch in range(0,args.max_epoch):
        for positive_sample,negative_sample,subsampling_weight,mode in train_iterator:
            log = kge_model.train_step(kge_model,optimizer,positive_sample,negative_sample,subsampling_weight,mode,args,True)
            training_logs.append(log)
        if epoch%args.log_epoch==0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
            log_metrics("evaluate on trainset during training",epoch,metrics)
            training_logs=[]
        if epoch%args.valid_epoch==0:
            metrics = kge_model.test_step(kge_model,valid_dataset_list,args)
            log_metrics("evaluate on validset during training",epoch,metrics)
            if metrics["MRR"]>best_mrr:
                best_mrr = metrics["MRR"]
                bad_epoch = 0
                save_variable_list = {
                    "epoch":epoch
                }
                save_model(kge_model,optimizer,save_variable_list,args,args.save_dir)
            else:
                bad_epoch += 1
        if bad_epoch>=args.early_stop_epoch:
            break
    checkpoint = torch.load(os.path.join(args.save_dir,"checkpoint"))
    best_epoch = checkpoint["epoch"]
    kge_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    test_logs = []
    for i in range(args.client_num):
        metrics = kge_model.test_step(kge_model,test_dataset_lists[i],args)
        test_logs.append(metrics)
        print("evaluate on testset %i"%(i))
        log_metrics("evaluate on testset after training",best_epoch,metrics)
    print("Weighted Averaged Results of All Clients")
    for metric in test_logs[0].keys():
        if metric!="n":
            print("%s: %f"%(metric,sum([test_log[metric]*test_log["n"] for test_log in test_logs])/sum([test_log["n"] for test_log in test_logs])))

if __name__=="__main__":
    main(parse_args())
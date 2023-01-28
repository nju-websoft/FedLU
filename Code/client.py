import argparse
import os
import json
import torch
import random
from copy import deepcopy
import logging
logging.basicConfig(level=logging.DEBUG)
import torch.nn as nn
import numpy as np
import Levenshtein
from model import KGEModel
from itertools import chain

from model import KGEModel
from dataloader import TrainDataset,TestDataset
from torch.utils.data import DataLoader

seed = 1998111213
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def _init_fn(worker_id):
    np.random.seed(int(seed))

class Client:
    def __init__(self,seq,args):
        self.name = seq
        print("args of client",self.name,args)
        if(args.local_file_dir is None):
            raise ValueError("local_file_dir should be set")
        self.local_file_dir = os.path.join(args.local_file_dir,str(seq))
        self.args = deepcopy(args)
        self.nrelation = self.load_relations()
        self.nentity = self.load_entities()
        self.load_dataset()

    def init_model(self,init_entity_embedding=None):
        args = self.args
        self.kgeModel = KGEModel(
            args.model,
            self.nentity,
            self.nrelation,
            args.hidden_dim,
            args.gamma,
            epsilon=args.epsilon,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding,
            entity_embedding=init_entity_embedding,
            fed_mode=args.fed_mode,
            eta=args.eta,
        )
        print("Parameters of client %i"%(self.name))
        for name,param in self.kgeModel.named_parameters():
            print("Parameter %s: %s, require_grad=%s"%(name,str(param.size()),str(param.requires_grad)))
        if(args.cuda):
            self.kgeModel = self.kgeModel.cuda()
        all_true_triples = self.traindata+self.validdata+self.testdata
        train_dataloader_head = DataLoader(
            TrainDataset(self.traindata,self.nentity,self.nrelation,args.negative_sample_size,"head-batch"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(self.traindata,self.nentity,self.nrelation,args.negative_sample_size,"tail-batch"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        train_iterator = list(chain.from_iterable(zip(train_dataloader_head,train_dataloader_tail)))
        self.train_iterator = train_iterator
        learning_rate = args.learning_rate
        self.optimizer = torch.optim.Adam(
            [{"params":self.kgeModel.entity_embedding},{"params":self.kgeModel.relation_embedding}],
            lr=args.learning_rate,
        )
        if args.fed_mode=="FedDist" and args.co_dist:
            self.fixoptimizer = torch.optim.Adam(
                [{"params":self.kgeModel.fixed_entity_embedding}],
                lr=args.learning_rate
            )
        valid_dataloader_head = DataLoader(
            TestDataset(
                self.validdata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "head-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        valid_dataloader_tail = DataLoader(
            TestDataset(
                self.validdata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "tail-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        self.valid_dataset_list = [valid_dataloader_head, valid_dataloader_tail]
        test_dataloader_head = DataLoader(
            TestDataset(
                self.testdata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "head-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        test_dataloader_tail = DataLoader(
            TestDataset(
                self.testdata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "tail-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        self.test_dataset_list = [test_dataloader_head,test_dataloader_tail]
        unlearn_path = os.path.join(self.local_file_dir,"unlearn.txt")
        self.unlearndata = self.read_triples(unlearn_path)
        self.retaindata = list(set(self.traindata)-set(self.unlearndata))
        all_true_triples = list(set(self.traindata+self.validdata+self.testdata)-set(self.unlearndata))
        unlearn_dataloader_head = DataLoader(
            TrainDataset(self.unlearndata,self.nentity,self.nrelation,args.negative_sample_size,"head-batch"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        unlearn_dataloader_tail = DataLoader(
            TrainDataset(self.unlearndata,self.nentity,self.nrelation,args.negative_sample_size,"tail-batch"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        self.unlearn_iterator = list(chain.from_iterable(zip(unlearn_dataloader_head,unlearn_dataloader_tail)))
        retain_dataloader_head = DataLoader(
            TrainDataset(self.retaindata,self.nentity,self.nrelation,args.negative_sample_size,"head-batch"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        retain_dataloader_tail = DataLoader(
            TrainDataset(self.retaindata,self.nentity,self.nrelation,args.negative_sample_size,"tail-batch"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        self.retain_iterator = list(chain.from_iterable(zip(retain_dataloader_head,retain_dataloader_tail)))
        unlearn_test_dataloader_head = DataLoader(
            TestDataset(
                self.unlearndata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "head-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        unlearn_test_dataloader_tail = DataLoader(
            TestDataset(
                self.unlearndata,
                # self.traindata+self.validdata+self.testdata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "tail-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        self.unlearn_test_dataset_list = [unlearn_test_dataloader_head,unlearn_test_dataloader_tail]
        valid_dataloader_head = DataLoader(
            TestDataset(
                self.validdata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "head-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        valid_dataloader_tail = DataLoader(
            TestDataset(
                self.validdata,
                all_true_triples,
                self.nentity,
                self.nrelation,
                "tail-batch",
            ),
            batch_size=args.test_batch_size,
            num_workers=max(0,args.cpu_num//2),
            collate_fn=TestDataset.collate_fn,
            worker_init_fn=_init_fn,
        )
        self.valid_dataset_list = [valid_dataloader_head, valid_dataloader_tail]
        
    def update_model(self,entity_embedding=None):
        if(entity_embedding is None):
            return
        else:
            self.kgeModel.update_model_embedding(entity_embedding)
            args = self.args
            if args.eta!=0.0:
                self.optimizer = torch.optim.Adam(
                    [{"params":self.kgeModel.entity_embedding},{"params":self.kgeModel.relation_embedding}],
                    lr=args.learning_rate,
                )
            if args.fed_mode=="FedDist" and args.co_dist:
                self.fixoptimizer = torch.optim.Adam(
                    [{"params":self.kgeModel.fixed_entity_embedding}],
                    lr=args.learning_rate
                )

    def load_entities(self):
        args = self.args
        entity2id = dict()
        entity2label = dict()
        with open(os.path.join(self.local_file_dir,"entities.dict"),"r",encoding="utf-8") as fin:
            for line in fin.readlines():
                line_split = line.strip().split()
                if(len(line_split)<2):
                    continue
                elif(len(line_split)==2):
                    id,entity = line_split[0],line_split[1]
                    label = entity
                else:
                    continue
                entity2id[entity] = int(id)
                entity2label[entity] = label
        self.entity2id = entity2id
        self.entity2label = entity2label
        return len(self.entity2id)

    def load_relations(self):
        args = self.args
        relation2id = dict()
        relation2label = dict()
        with open(os.path.join(self.local_file_dir,"relations.dict"),"r",encoding="utf-8") as fin:
            for line in fin.readlines():
                line_split = line.strip().split()
                if(len(line_split)<2):
                    continue
                elif(len(line_split)==2):
                    id,relation = line_split[0],line_split[1]
                    label = relation
                else:
                    continue
                relation2id[relation] = int(id)
                relation2label[relation] = label
        self.relation2id = relation2id
        self.relation2label = relation2label
        return len(self.relation2id)

    def read_triples(self,file_path):
        triples = []
        with open(file_path,"r",encoding="utf-8") as fin:
            for line in fin.readlines():
                h,r,t = line.strip().split()
                triples.append((self.entity2id[h],self.relation2id[r],self.entity2id[t]))
        return triples

    def load_dataset(self):
        args = self.args
        train_path = os.path.join(self.local_file_dir,"train.txt")
        self.traindata = self.read_triples(train_path)
        valid_path = os.path.join(self.local_file_dir,"valid.txt")
        self.validdata = self.read_triples(valid_path)
        test_path = os.path.join(self.local_file_dir,"test.txt")
        self.testdata = self.read_triples(test_path)

    def train(self,nodistg2l=True,nodistl2g=True):
        args = self.args
        training_logs = []
        transfer_logs = []
        for epoch in range(0,args.max_epoch):
            for positive_sample,negative_sample,subsampling_weight,mode in self.train_iterator:
                training_log = self.kgeModel.train_step(self.kgeModel,self.optimizer,positive_sample,negative_sample,subsampling_weight,mode,args,nodistg2l)
                training_logs.append(training_log)
                if args.fed_mode=="FedDist":
                    transfer_log = self.kgeModel.transfer_step(self.kgeModel,self.fixoptimizer,positive_sample,negative_sample,subsampling_weight,mode,args,nodistl2g)
                    transfer_logs.append(transfer_log)
        metrics = {}
        if len(training_logs)!=0:
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                print("%s: %f"%(metric,metrics[metric]))  
        if len(transfer_logs)!=0:
            for metric in transfer_logs[0].keys():
                print("tranfer %s:%f"%(metric,sum([log[metric] for log in transfer_logs])/len(transfer_logs)))
        return metrics

    def train_round(self):
        args = self.args
        training_logs = []
        bad_epoch = 0
        best_mrr = 0
        best_epoch = 0
        print("client %i log args during train round: %s" %(self.name,str(args)))
        for epoch in range(0,args.max_epoch):
            for positive_sample,negative_sample,subsampling_weight,mode in self.train_iterator:
                tmplog = self.kgeModel.train_step(self.kgeModel,self.optimizer,positive_sample,negative_sample,subsampling_weight,mode,args,True)
                training_logs.append(tmplog)
            if epoch%args.log_epoch==0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics("evaluate on trainset during training",epoch,metrics)
                training_logs=[]
            if epoch%args.valid_epoch==0:
                metrics = self.valid()
                log_metrics("evaluate on validset during training",epoch,metrics)
                if metrics["MRR"]>best_mrr:
                    best_mrr = metrics["MRR"]
                    bad_epoch=0
                    best_epoch = epoch
                    save_variable_list={
                        "epoch":best_epoch
                    }
                    self.save_model(save_variable_list)
                else:
                    bad_epoch += 1
            if bad_epoch>=args.early_stop_epoch:
                break
        if (epoch%args.valid_epoch!=0) and (bad_epoch<args.early_stop_epoch):
            metrics = self.valid()
            log_metrics("evaluate on validset during training",epoch,metrics)
            if metrics["MRR"]>best_mrr:
                best_mrr = metrics["MRR"]
                best_epoch = epoch
                save_variable_list={
                    "epoch":best_epoch
                }
                self.save_model(save_variable_list)
        self.load_model()
        metrics = self.test()
        log_metrics("evaluate on testset after training",best_epoch,metrics)
        return metrics

    def save_model(self,save_variable_list):
        args = self.args
        save_path = os.path.join(args.save_dir,str(self.name))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        argparse_dict = vars(args)
        with open(os.path.join(save_path,"config.json"),"w",encoding="utf-8") as fjson:
            json.dump(argparse_dict,fjson)
        if args.fed_mode=="FedDist" and args.co_dist:
            fixoptimizer_state_dict={"fixoptimizer_state_dict":self.fixoptimizer.state_dict()}
        else:
            fixoptimizer_state_dict = {}
        torch.save(
            {
                **save_variable_list,
                "model_state_dict":self.kgeModel.state_dict(),
                "optimizer_state_dict":self.optimizer.state_dict(),
                **fixoptimizer_state_dict
            },
            os.path.join(save_path,"checkpoint")
        )
        entity_embedding = self.kgeModel.entity_embedding.detach().cpu().numpy()
        np.save(os.path.join(save_path,"entity_embedding"),entity_embedding)
        relation_embedding = self.kgeModel.relation_embedding.detach().cpu().numpy()
        np.save(os.path.join(save_path,"relation_embedding"),relation_embedding)
        if args.fed_mode=="FedDist":
            fixed_entity_embedding = self.kgeModel.fixed_entity_embedding.detach().cpu().numpy()
            np.save(os.path.join(save_path,"fixed_entity_embedding"),fixed_entity_embedding)
        return

    def load_model(self):
        args = self.args
        save_path = os.path.join(args.save_dir,str(self.name))
        checkpoint = torch.load(os.path.join(save_path,"checkpoint"))

        self.kgeModel.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if args.fed_mode=="FedDist" and args.co_dist:
            self.fixoptimizer.load_state_dict(checkpoint["fixoptimizer_state_dict"])

    def get_entity_embedding(self):
        args = self.args
        if args.fed_mode in ["FedDist"]:
            return self.kgeModel.fixed_entity_embedding
        else:
            return self.kgeModel.entity_embedding

    def valid(self):
        args = self.args
        if args.fed_mode in ["FedDist"]:
            return self.kgeModel.test_step(self.kgeModel,self.valid_dataset_list,self.args),self.kgeModel.global_test(self.kgeModel,self.valid_dataset_list,self.args)
        else:
            return self.kgeModel.test_step(self.kgeModel,self.valid_dataset_list,self.args)
    def test(self):
        return self.kgeModel.test_step(self.kgeModel,self.test_dataset_list,self.args)

    def unlearn(self):
        args = self.args
        self.kgeModel.relation_embedding.requires_grad = False
        self.optimizer = torch.optim.Adam(
            [{"params":self.kgeModel.entity_embedding}],
            lr=args.learning_rate,
        )
        self.fixoptimizer = torch.optim.Adam(
            [{"params":self.kgeModel.fixed_entity_embedding}],
            lr=args.learning_rate,
        )
        for epoch in range(0,args.max_unlearn_epoch):
            for unlearn_sample,negative_sample,subsampling_weight,mode in self.unlearn_iterator:
                global_unlearn_log = self.kgeModel.unlearn_propagate(self.kgeModel,self.fixoptimizer,unlearn_sample,negative_sample,subsampling_weight,mode,args)
            for unlearn_sample,negative_sample,subsampling_weight,mode in self.unlearn_iterator:
                unlearn_log = self.kgeModel.unlearn_step(self.kgeModel,self.optimizer,unlearn_sample,negative_sample,subsampling_weight,mode,args)
            
            for unlearn_sample,negative_sample,subsampling_weight,mode in self.unlearn_iterator:
                global_unlearn_log = self.kgeModel.unlearn_propagate(self.kgeModel,self.fixoptimizer,unlearn_sample,negative_sample,subsampling_weight,mode,args)
                unlearn_log = self.kgeModel.unlearn_step(self.kgeModel,self.optimizer,unlearn_sample,negative_sample,subsampling_weight,mode,args)

        best_mrr = 0.0
        for epoch in range(0,args.max_unlearn_epoch):
            for retain_sample,negative_sample,subsampling_weight,mode in self.retain_iterator:
                self.kgeModel.train_step(self.kgeModel,self.optimizer,retain_sample,negative_sample,subsampling_weight,mode,args,nodist=True)
                self.kgeModel.transfer_step(self.kgeModel,self.fixoptimizer,retain_sample,negative_sample,subsampling_weight,mode,args,nodist=True)
                global_retain_log = self.kgeModel.transfer_step(self.kgeModel,self.fixoptimizer,retain_sample,negative_sample,subsampling_weight,mode,args,nodist=False)
                self.kgeModel.transfer_step(self.kgeModel,self.fixoptimizer,retain_sample,negative_sample,subsampling_weight,mode,args,nodist=True)
                local_retain_log = self.kgeModel.train_step(self.kgeModel,self.optimizer,retain_sample,negative_sample,subsampling_weight,mode,args,nodist=False)
            valid_metrics = self.kgeModel.test_step(self.kgeModel,self.valid_dataset_list,self.args)
            if valid_metrics["MRR"]>best_mrr:
                best_mrr = valid_metrics["MRR"]
            else:
                break
    def unlearn_test(self):
        args = self.args
        args = self.args
        local_unlearn_metrics = self.kgeModel.test_step(self.kgeModel,self.unlearn_test_dataset_list,self.args)
        local_test_metrics = self.kgeModel.test_step(self.kgeModel,self.test_dataset_list,self.args)
        global_unlearn_metrics = self.kgeModel.global_test(self.kgeModel,self.unlearn_test_dataset_list,self.args)
        global_test_metrics = self.kgeModel.global_test(self.kgeModel,self.test_dataset_list,self.args)
        return local_unlearn_metrics,local_test_metrics,global_unlearn_metrics,global_test_metrics

    def retrain(self):
        args = self.args
        self.retrainModel = KGEModel(
            args.model,
            self.nentity,
            self.nrelation,
            args.hidden_dim,
            args.gamma,
            epsilon=args.epsilon,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding,
        )
        for name,param in self.retrainModel.named_parameters():
            print("Parameter %s: %s, require_grad=%s"%(name,str(param.size()),str(param.requires_grad)))
        if(args.cuda):
            self.retrainModel = self.retrainModel.cuda()
        self.retrainoptimizer = torch.optim.Adam(
            [{"params":self.retrainModel.entity_embedding},{"params":self.retrainModel.relation_embedding}],
            lr = args.learning_rate
        )
        training_logs = []
        best_mrr = 0
        best_epoch = 0
        for epoch in range(0,args.max_retrain_epoch):
            for retain_sample,negative_sample,subsampling_weight,mode in self.retain_iterator:
                training_log = self.kgeModel.train_step(self.retrainModel,self.retrainoptimizer,retain_sample,negative_sample,subsampling_weight,mode,args,nodist=True)
                training_logs.append(training_log)
            if epoch%3==0:
                valid_metrics = self.retrainModel.test_step(self.retrainModel,self.valid_dataset_list,args)
                if valid_metrics["MRR"]>best_mrr:
                    best_mrr = valid_metrics["MRR"]
                    best_epoch = epoch
                    test_metrics = self.retrainModel.test_step(self.retrainModel,self.test_dataset_list,args)
                    log_metrics("retrain test",epoch,test_metrics)
                    unlearn_metrics = self.retrainModel.test_step(self.retrainModel,self.unlearn_test_dataset_list,args)
                    log_metrics("retrain unlearn",epoch,unlearn_metrics)
                else:
                    break
        return unlearn_metrics,test_metrics
            
def log_metrics(mode,epoch,metrics):
    print("Mode: %s"%(mode))
    print("Epoch: %i"%(epoch))
    for metric in metrics.keys():
        print("%s: %f"%(metric,metrics[metric]))
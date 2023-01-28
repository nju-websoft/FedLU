from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from server import Server
from client import Client
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

class UnlearnController:
    def __init__(self):
        return
    
    def parse_args(self,args=None):
        parser = argparse.ArgumentParser(
            description="train, valid, test and unlearn kge models collaboratively",
            usage = "controller.py [<args>] [-h | --help]"
        )
        parser.add_argument("--client_num",type=int,default=3,help="client num, type int, default 3")
        parser.add_argument("--local_file_dir",type=str,default="../Data/FB15k-237/C3FL",help="local file directory, type str, default ../Data/FB15k-237/C3FL")
        parser.add_argument("--save_dir",type=str,default="../Output/FB15k-237/C3FL",help="save dir, type str, default ../Output/FB15k-237/C3FL")
        parser.add_argument("--aggregate_iteration",type=int,default=200,help="aggregate iterations, type int, default 200")
        parser.add_argument("--cuda",action="store_true",help="use GPU, store true")
        parser.add_argument("--model",type=str,default="TransE",help="model, type str, choose between TransE/DisMult/ComplEx/RotatE")
        parser.add_argument("--double_entity_embedding",action="store_true",help="double entity embedding, store true")
        parser.add_argument("--double_relation_embedding",action="store_true",help="double relation embedding, store true")
        parser.add_argument("--max_epoch",type=int,default=400,help="max epoch, type int, default 400")
        parser.add_argument("--valid_epoch",type=int,default=10,help="valid epoch, type int, default 10")
        parser.add_argument("--early_stop_epoch",type=int,default=15,help="early stop epoch, type int, default 15")
        parser.add_argument("--cpu_num",type=int,default=16,help="cpu num, type int, default 16")
        parser.add_argument("--negative_sample_size",type=int,default=256,help="negative sample size, type int, default 256")
        parser.add_argument("--negative_adversarial_sampling",action="store_true",help="negative adversarial sampling, store true")
        parser.add_argument("--adversarial_temperature",type=float,default=1.0,help="float, adversarial temperature, default 1.0")
        parser.add_argument("--uni_weight",action="store_true",help="uni weight, store true")
        parser.add_argument("--regularization",type=float,default=0.0,help="regularization, type float, default 0.0")
        parser.add_argument("--batch_size",type=int,default=1024,help="batch size, type int, default 1024")
        parser.add_argument("--hidden_dim",type=int,default=256,help="hidden dim, type int, default 256")
        parser.add_argument("--learning_rate",type=float,default=1e-4,help="learning rate, type float, default 1e-4")
        parser.add_argument("--gamma",type=float,default=12.0,help="gamma, type float, default 9.0")
        parser.add_argument("--epsilon",type=float,default=2.0,help="epsilon, type float, default 2.0")
        parser.add_argument("--test_batch_size",type=int,default=64,help="test batch size, type int, default 32")
        parser.add_argument("--log_epoch",type=int,default=10,help="log epoch, type int, default 10")
        parser.add_argument("--test_log_step",type=int,default=200,help="test log step, type int, default 200")
        parser.add_argument("--fed_mode",type=str,default="FedAvg",help="fed mode, type str, choose from FedAvg/FedProx/FedDist")
        parser.add_argument("--mu",type=float,default=0.0,help="mu, type float, default 0.0")
        parser.add_argument("--mu_decay",action="store_true",help="mu decay, store true")
        parser.add_argument("--mu_single_entity",action="store_true",help="mu single entity, store true")
        parser.add_argument("--eta",type=float,default=1.0,help="eta, type float, default 1.0")
        parser.add_argument("--agg",type=str,default="weighted",help="aggregation method, type str, default weighted, optional weighted/distance/similarity")
        parser.add_argument("--max_iter",type=int,default=300,help="max iter, type int, default 300")
        parser.add_argument("--valid_iter",type=int,default=5,help="valid iter, type int, default 5")
        parser.add_argument("--early_stop_iter",type=int,default=15,help="early stop iter, type int, default 15")
        parser.add_argument("--dist_mu",type=float,default=1e-2,help="distillation mu, type float, default 1e-2")
        parser.add_argument("--co_dist",action="store_true",help="co-distillation, store true")
        parser.add_argument("--wait_iter",type=int,default=10)
        parser.add_argument("--max_unlearn_epoch",type=int,default=10)
        parser.add_argument("--confusion_mu",type=float,default=1e-2)
        parser.add_argument("--max_retrain_epoch",type=int,default=200)
        args = parser.parse_args(args)
        if args.local_file_dir is None:
            raise ValueError("local file dir must be set")
        if args.fed_mode=="FedDist":
            args.eta=0.0
        elif args.fed_mode=="FedAvg":
            args.eta=1.0
        if args.model=="RotatE":
            args.double_entity_embedding=True
            args.negative_adversarial_sampling=True
        elif args.model=="ComplEx":
            args.double_entity_embedding=True
            args.double_relation_embedding=True
        self.args = args
    
    def init_federation(self):
        args = self.args
        server = Server(args)
        clients = []
        for i in range(args.client_num):
            client = Client(i,args)
            clients.append(client)
        self.server = server
        self.clients = clients
    
    def init_model(self):
        args = self.args
        self.server.generate_global_embedding()
        client_embedding_dict = self.server.assign_embedding()
        for i in range(args.client_num):
            self.clients[i].init_model(init_entity_embedding=client_embedding_dict[i])
        
    def save(self):
        args = self.args
        for i in range(0,args.client_num):
            self.clients[i].save_model({})
    def load(self):
        args = self.args
        for i in range(0,args.client_num):
            self.clients[i].load_model()

    def pipeline(self):
        args = self.args
        self.load()
        raw_local_unlearn_logs = []
        raw_local_test_logs = []
        raw_global_unlearn_logs = []
        raw_global_test_logs = []
        local_unlearn_logs = []
        local_test_logs = []
        global_unlearn_logs = []
        global_test_logs = []
        for i in range(0,args.client_num):
            raw_local_unlearn_log,raw_local_test_log,raw_global_unlearn_log,raw_global_test_log = self.clients[i].unlearn_test()
            raw_local_unlearn_logs.append(raw_local_unlearn_log)
            raw_local_test_logs.append(raw_local_test_log)
            raw_global_unlearn_logs.append(raw_global_unlearn_log)
            raw_global_test_logs.append(raw_global_test_log)

            self.clients[i].unlearn()
            local_unlearn_log,local_test_log,global_unlearn_log,global_test_log = self.clients[i].unlearn_test()
            local_unlearn_logs.append(local_unlearn_log)
            local_test_logs.append(local_test_log)
            global_unlearn_logs.append(global_unlearn_log)
            global_test_logs.append(global_test_log)

        log_metrics("raw local unlearn",0,raw_local_unlearn_logs)
        log_metrics("raw local test",0,raw_local_test_logs)
        log_metrics("raw global unlearn",0,raw_global_unlearn_logs)
        log_metrics("raw global test",0,raw_global_test_logs)
        log_metrics("update local unlearn",0,local_unlearn_logs)
        log_metrics("update local test",0,local_test_logs)
        log_metrics("update global unlearn",0,global_unlearn_logs)
        log_metrics("update global test",0,global_test_logs)
                
def log_metrics(mode,iter,logs):
    print("-"*20+"\n")
    print("%s in Iter %i"%(mode,iter))
    for i in range(0,len(logs)):
        print("Log of Client %i"%(i))
        for metric in logs[i].keys():
            if metric!="n":
                print("%s:%f"%(metric,logs[i][metric]))
    print("Weight Average and Variance of All Clients")
    for metric in logs[0].keys():
        if metric!="n":
            weighted_metric = sum([log[metric]*log["n"] for log in logs])/sum([log["n"] for log in logs])
            weighted_variance = sum([log["n"]*(log[metric]-weighted_metric)**2 for log in logs])/sum([log["n"] for log in logs])
            print("%s: %f, %f"%(metric,weighted_metric,weighted_variance))

if __name__=="__main__":
    controller = UnlearnController()
    controller.parse_args()
    controller.init_federation()
    controller.init_model()
    controller.pipeline()
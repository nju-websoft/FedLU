import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import dataloader
from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self,model_name,nentity,nrelation,hidden_dim,gamma,epsilon=2.0,
                double_entity_embedding=False,double_relation_embedding=False,
                entity_embedding=None,fed_mode="FedAvg",eta=1.0):
        super(KGEModel,self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item()+self.epsilon)/hidden_dim]),
            requires_grad=False
        )
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.fed_mode = fed_mode
        self.eta = eta
        # receive entity and relation embedding feed in by server
        if (entity_embedding is not None):
            if self.fed_mode in ["FedEC"]:
                self.old_entity_embedding = nn.Parameter(entity_embedding.clone().cuda(),requires_grad=False)
            self.entity_embedding = nn.Parameter(entity_embedding)
            if self.fed_mode in["FedProx","FedProx","FedEC"]:
                self.fixed_entity_embedding = nn.Parameter(entity_embedding.clone().cuda(),requires_grad=False)
            elif self.fed_mode in ["FedDist"]:
                self.fixed_entity_embedding = nn.Parameter(entity_embedding.clone().cuda(),requires_grad=True)
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(self.nentity,self.entity_dim))
            nn.init.uniform_(
                tensor = self.entity_embedding,
                a = -self.embedding_range.item(),
                b = self.embedding_range.item()
            )

        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation,self.relation_dim))
        nn.init.uniform_(
            tensor = self.relation_embedding,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item()
        )
        if (model_name=="pRotatE"):
            self.modulus = nn.Parameter(torch.Tensor([[0.5*self.embedding_range.item()]]))

        if (model_name not in ["TransE","DisMult","ComplEx","RotatE","pRotatE"]):
            raise ValueError("model %s not support" %model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')
        if (model_name=="ComplEx" and not(double_entity_embedding and double_relation_embedding)):
            raise ValueError("ComplEx should use double_entity_embedding and double_relation_embedding")

    def update_model_embedding(self,entity_embedding=None):
        if(entity_embedding is not None):
            if self.fed_mode in ["FedEC"]:
                self.old_entity_embedding = nn.Parameter(self.entity_embedding.clone().cuda(),requires_grad=False)
            if self.eta==1.0:
                self.entity_embedding = nn.Parameter(entity_embedding.cuda(),requires_grad=True)
            elif self.eta!=0.0:
                self.entity_embedding = nn.Parameter(self.eta*entity_embedding.cuda()+(1.0-self.eta)*self.entity_embedding,requires_grad=True)
            if self.fed_mode in ["FedProx","FedEC"]:
                self.fixed_entity_embedding = nn.Parameter(entity_embedding.clone().cuda(),requires_grad=False)
            elif self.fed_mode in ["FedDist"]:
                self.fixed_entity_embedding = nn.Parameter(entity_embedding.clone().cuda(),requires_grad=True)

    def forward(self,sample,mode="single"):
        if(mode=="single"):
            batch_size, negative_sample_size = sample.size(0),1
            relation = torch.index_select(
                self.relation_embedding,
                dim = 0,
                index = sample[:,1]
            ).unsqueeze(1)
            head = torch.index_select(
                self.entity_embedding,
                dim = 0,
                index = sample[:,0]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                 dim = 0,
                 index = sample[:,2]
            ).unsqueeze(1)
        elif(mode=="head-batch"):
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim = 0,
                index = head_part.view(-1)
            ).view(batch_size,negative_sample_size,-1)
            relation = torch.index_select(
                self.relation_embedding,
                 dim = 0,
                 index = tail_part[:,1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                dim = 0,
                index = tail_part[:,2]
            ).unsqueeze(1)
        elif(mode=="tail-batch"):
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim = 0,
                index = head_part[:,0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim = 0,
                index = head_part[:,1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.entity_embedding,
                dim = 0,
                index = tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError("mode %s not supported" %mode)

        model_func = {
            "TransE": self.TransE,
            "DisMult": self.DistMult,
            "ComplEx": self.ComplEx,
            "RotatE": self.RotatE,
            "pRotatE": self.pRotatE
        }
        if(self.model_name in model_func):
            score = model_func[self.model_name](head,relation,tail,mode)
        else:
            raise ValueError("model %s not supported" %self.model_name)
        return score

    def global_score(self,sample,mode="single"):
        if(mode=="single"):
            batch_size, negative_sample_size = sample.size(0),1
            relation = torch.index_select(
                self.relation_embedding,
                dim = 0,
                index = sample[:,1]
            ).unsqueeze(1)
            head = torch.index_select(
                self.fixed_entity_embedding,
                dim = 0,
                index = sample[:,0]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.fixed_entity_embedding,
                 dim = 0,
                 index = sample[:,2]
            ).unsqueeze(1)
        elif(mode=="head-batch"):
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = torch.index_select(
                self.fixed_entity_embedding,
                dim = 0,
                index = head_part.view(-1)
            ).view(batch_size,negative_sample_size,-1)
            relation = torch.index_select(
                self.relation_embedding,
                 dim = 0,
                 index = tail_part[:,1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.fixed_entity_embedding,
                dim = 0,
                index = tail_part[:,2]
            ).unsqueeze(1)
        elif(mode=="tail-batch"):
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.fixed_entity_embedding,
                dim = 0,
                index = head_part[:,0]
            ).unsqueeze(1)
            relation = torch.index_select(
                self.relation_embedding,
                dim = 0,
                index = head_part[:,1]
            ).unsqueeze(1)
            tail = torch.index_select(
                self.fixed_entity_embedding,
                dim = 0,
                index = tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError("mode %s not supported" %mode)

        model_func = {
            "TransE": self.TransE,
            "DisMult": self.DistMult,
            "ComplEx": self.ComplEx,
            "RotatE": self.RotatE,
            "pRotatE": self.pRotatE
        }
        if(self.model_name in model_func):
            score = model_func[self.model_name](head,relation,tail,mode)
        else:
            raise ValueError("model %s not supported" %self.model_name)
        return score

    def TransE(self,head,relation,tail,mode):
        if(mode=="head-batch"):
            score = head + (relation-tail)
        else:
            score = (head+relation) - tail
        score = self.gamma.item() - torch.norm(score,p=1,dim=2)
        return score

    def DistMult(self,head,relation,tail,mode):
        if(mode=="head-batch"):
            score = head*(relation*tail)
        else:
            score = (head*relation)*tail
        score = score.sum(dim=2)
        return score

    def ComplEx(self,head,relation,tail,mode):
        re_head, im_head = torch.chunk(head,2,dim=2)
        re_relation, im_relation = torch.chunk(relation,2,dim=2)
        re_tail, im_tail = torch.chunk(tail,2,dim=2)
        if(mode=="head-batch"):
            re_score = re_relation*re_tail+im_relation*im_tail
            im_score = re_relation*im_tail-im_relation*re_tail
            score = re_head*re_score+im_head*im_score
        else:
            re_score = re_head*re_relation-im_head*im_relation
            im_score = re_head*im_relation+im_head*re_relation
            score = re_score*re_tail+im_score*im_tail
        score = score.sum(dim=2)
        return score

    def RotatE(self,head,relation,tail,mode):
        pi = 3.14159265358979323846
        re_head,im_head = torch.chunk(head,2,dim=2)
        re_tail,im_tail = torch.chunk(tail,2,dim=2)
        phase_relation = relation/(self.embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if(mode=="head-batch"):
            re_score = re_relation*re_tail+im_relation*im_tail
            im_score = re_relation*im_tail-im_relation*re_tail
            re_score = re_score-re_head
            im_score = im_score-im_head
        else:
            re_score = re_head*re_relation-im_head*im_relation
            im_score = re_head*im_relation+im_head*re_relation
            re_score = re_score-re_tail
            im_score = im_score-im_tail
        score = torch.stack([re_score,im_score],dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item()-score.sum(dim=2)
        return score

    def pRotatE(self,head,relation,tail,mode):
        pi=3.14159265358979323846
        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)
        if(mode=="head-batch"):
            score = phase_head+(phase_relation-phase_tail)
        else:
            score = (phase_head+phase_relation)-phase_tail
        score = torch.sin(score)
        score = torch.abs(score)
        score = self.gamma.item()-score.sum(dim=2)*self.modulus
        return score
    
    @staticmethod
    def train_step(model,optimizer,positive_sample,negative_sample,subsampling_weight,mode,args,nodist=True):
        model.train()
        optimizer.zero_grad()
        if(args.cuda):
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        negative_score = model((positive_sample,negative_sample), mode=mode)
        if args.fed_mode=="FedDist" and not nodist:
            local_prob = negative_score.clone()
            global_prob = model.global_score((positive_sample,negative_sample),mode=mode)

        if(args.negative_adversarial_sampling):
            negative_score = (F.softmax(negative_score*args.adversarial_temperature,dim=1).detach()
                                * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        positive_score = model(positive_sample)

        if args.fed_mode=="FedDist" and args.dist_mu!=0.0 and not nodist:
            local_prob = torch.cat((local_prob,positive_score.clone()),dim=1)
            local_prob = F.log_softmax(local_prob,dim=1)
            global_pos = model.global_score(positive_sample)
            global_prob = torch.cat((global_prob,global_pos),dim=1)
            global_prob = F.softmax(global_prob,dim=1)
            loss_KD = nn.KLDivLoss(reduction="batchmean")
            distill_loss = args.dist_mu*loss_KD(local_prob,global_prob.detach())

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if(args.uni_weight):
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(subsampling_weight*positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight*negative_score).sum()/subsampling_weight.sum()
        loss = (positive_sample_loss+negative_sample_loss)/2

        if args.fed_mode=="FedDist" and args.dist_mu!=0.0 and not nodist:
            loss += distill_loss
            distill_log = {"distillation":distill_loss.item()}
        else:
            distill_log = {}

        if(args.regularization!=0.0):
            regularization = args.regularization*(
                model.entity_embedding.norm(p=3)**3+
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss+regularization
            regularization_log = {"regularization":regularization.item()}
        else:
            regularization_log = {}

        if(args.fed_mode=="FedProx" and args.mu!=0.0):
            entity_regularization = 0.5*args.mu*(
                (model.entity_embedding-model.fixed_entity_embedding).norm(p=2)**2
            )
            if args.mu_single_entity:
                entity_regularization = entity_regularization.mean()
            else:
                entity_regularization = entity_regularization.sum()
            loss = loss+entity_regularization
            entity_regularization_log = {"entity_regularization":entity_regularization.item()}
        else:
            entity_regularization_log = {}
        
        if args.fed_mode=="FedEC":
            sim=nn.CosineSimilarity(dim=-1)
            simLocal = sim(model.entity_embedding,model.old_entity_embedding).mean()/args.mu_temperature
            simGlobal = sim(model.entity_embedding,model.fixed_entity_embedding).mean()/args.mu_temperature
            simLocal = torch.exp(simLocal)
            simGlobal = torch.exp(simGlobal)
            contrastive_loss = -args.mu_contrastive*torch.log(simGlobal/(simGlobal+simLocal))
            loss = loss + contrastive_loss
            contrastive_loss_log = {"contrastive_loss":contrastive_loss}
        else:
            contrastive_loss_log = {}


        loss.backward()
        optimizer.step()
        log = {
            **regularization_log,
            **entity_regularization_log,
            **distill_log,
            **contrastive_loss_log,
            "positive_sample_loss": positive_sample_loss.item(),
            "negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item()
        }
        return log

    @staticmethod
    def transfer_step(model,optimizer,positive_sample,negative_sample,subsampling_weight,mode,args,nodist):
        model.train()
        optimizer.zero_grad()
        if(args.cuda):
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        negative_score = model.global_score((positive_sample,negative_sample),mode=mode)

        if not nodist:
            global_prob = negative_score.clone()
            local_prob = model((positive_sample,negative_sample), mode=mode)

        if(args.negative_adversarial_sampling):
            negative_score = (F.softmax(negative_score*args.adversarial_temperature,dim=1).detach()
                                * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        positive_score = model.global_score(positive_sample)

        if not nodist:
            global_prob = torch.cat((global_prob,positive_score.clone()),dim=1)
            global_prob = F.log_softmax(global_prob,dim=1)
            local_pos = model(positive_sample)
            local_prob = torch.cat((local_prob,local_pos),dim=1)
            local_prob = F.softmax(local_prob,dim=1)
            loss_KD = nn.KLDivLoss(reduction="batchmean")
            distill_loss = args.dist_mu*loss_KD(global_prob,local_prob.detach())

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if(args.uni_weight):
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(subsampling_weight*positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight*negative_score).sum()/subsampling_weight.sum()
        loss = (positive_sample_loss+negative_sample_loss)/2

        if not nodist:
            loss += distill_loss
            distill_log = {"distillation":distill_loss.item()}
        else:
            distill_log = {}

        if args.regularization!=0.0:
            regularization = args.regularization * (
                model.fixed_entity_embedding.norm(p=3)**3+
                model.relation_embedding.norm(p=3).norm(p=3)**3
            )
            loss = loss + regularization
            regularization_log = {"regularization":regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        optimizer.step()
        log = {
            **distill_log,
            **regularization_log,
            "global_positive_sample_loss": positive_sample_loss.item(),
            "global_negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item()
        }
        return log
    
    @staticmethod
    def test_step(model,test_dataset_list,args):
        model.eval()
        logs = []
        step=0
        total_steps=sum([len(dataset) for dataset in test_dataset_list])
        total_n = 0
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample,negative_sample,filter_bias,mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                        model.entity_embedding = model.entity_embedding.cuda()
                        model.relation_embedding = model.relation_embedding.cuda()
                    batch_size = positive_sample.size(0)
                    total_n += batch_size
                    score = model((positive_sample,negative_sample),mode)
                    score += filter_bias
                    argsort = torch.argsort(score,dim=1,descending=True)
                    if(mode=="head-batch"):
                        positive_arg=positive_sample[:,0]
                    elif mode=="tail-batch":
                        positive_arg=positive_sample[:,2]
                    else:
                        raise ValueError("mode %s not supported" %mode)
                    for i in range(batch_size):
                        ranking = (argsort[i,:]==positive_arg[i])
                        ranking = ranking.nonzero()
                        assert ranking.size(0)==1
                        ranking=1+ranking.item()
                        logs.append({
                            "MRR":1.0/ranking,
                            "MR":float(ranking),
                            "HITS@1":1.0 if ranking<=1 else 0.0,
                            "HITS@3":1.0 if ranking<=3 else 0.0,
                            "HITS@10":1.0 if ranking<=10 else 0.0,
                        })
                    step+=1
        metrics={}
        for metric in logs[0].keys():
            metrics[metric]=sum([log[metric] for log in logs])/len(logs)
        metrics["n"] = total_n
        return metrics

    @staticmethod
    def global_test(model,test_dataset_list,args):
        model.eval()
        logs = []
        step=0
        total_steps=sum([len(dataset) for dataset in test_dataset_list])
        total_n = 0
        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample,negative_sample,filter_bias,mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                        model.fixed_entity_embedding = model.fixed_entity_embedding.cuda()
                        model.relation_embedding = model.relation_embedding.cuda()
                    batch_size = positive_sample.size(0)
                    total_n += batch_size
                    score = model.global_score((positive_sample,negative_sample),mode)
                    score += filter_bias
                    argsort = torch.argsort(score,dim=1,descending=True)
                    if(mode=="head-batch"):
                        positive_arg=positive_sample[:,0]
                    elif mode=="tail-batch":
                        positive_arg=positive_sample[:,2]
                    else:
                        raise ValueError("mode %s not supported" %mode)
                    for i in range(batch_size):
                        ranking = (argsort[i,:]==positive_arg[i])
                        ranking = ranking.nonzero()
                        assert ranking.size(0)==1
                        ranking=1+ranking.item()
                        logs.append({
                            "MRR":1.0/ranking,
                            "MR":float(ranking),
                            "HITS@1":1.0 if ranking<=1 else 0.0,
                            "HITS@3":1.0 if ranking<=3 else 0.0,
                            "HITS@10":1.0 if ranking<=10 else 0.0,
                        })
                    step+=1
        metrics={}
        for metric in logs[0].keys():
            metrics[metric]=sum([log[metric] for log in logs])/len(logs)
        metrics["n"] = total_n
        return metrics

    @staticmethod
    def unlearn_step(model,optimizer,unlearn_sample,negative_sample,subsampling_weight,mode,args):
        model.train()
        optimizer.zero_grad()
        if(args.cuda):
            unlearn_sample = unlearn_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        negative_score = model((unlearn_sample,negative_sample), mode=mode)
        negative_prob = negative_score.clone()
        
        if(args.negative_adversarial_sampling):
            negative_score = (F.softmax(negative_score*args.adversarial_temperature,dim=1).detach()
                                * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        unlearn_score = model(unlearn_sample)
        unlearn_prob = unlearn_score.clone()

        distance_loss = negative_prob-unlearn_prob
        distance_loss = args.confusion_mu*distance_loss.norm(p=2,dim=1).mean()
        unlearn_score = F.logsigmoid(-unlearn_score).squeeze(dim=1)

        if(args.uni_weight):
            unlearn_sample_loss = -unlearn_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            unlearn_sample_loss = -(subsampling_weight*unlearn_score).sum()/subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight*negative_score).sum()/subsampling_weight.sum()

        loss = (10*unlearn_sample_loss+negative_sample_loss)/2
        loss += distance_loss
        loss.backward()
        optimizer.step()
        log = {
            "distance_loss":distance_loss.item(),
            "unlearn_sample_loss": unlearn_sample_loss.item(),
            "negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item()
        }
        return log
    
    @staticmethod
    def unlearn_propagate(model,optimizer,unlearn_sample,negative_sample,subsampling_weight,mode,args):
        model.train()
        optimizer.zero_grad()
        if(args.cuda):
            unlearn_sample = unlearn_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            
        global_negative_score = model.global_score((unlearn_sample,negative_sample),mode=mode)
        global_negative_prob = global_negative_score.clone()
        local_negative_prob = model((unlearn_sample,negative_sample), mode=mode)

        if(args.negative_adversarial_sampling):
            negative_score = (F.softmax(global_negative_score*args.adversarial_temperature,dim=1).detach()
                                * F.logsigmoid(-global_negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-global_negative_score).mean(dim=1)
        global_unlearn_score = model.global_score(unlearn_sample)
        global_unlearn_prob = global_unlearn_score.clone()

        distance_loss = global_negative_prob-global_unlearn_prob
        distance_loss = args.confusion_mu*distance_loss.norm(p=2,dim=1).mean()
        global_prob = torch.cat((global_negative_prob,global_unlearn_prob),dim=1)
        global_prob = F.log_softmax(global_prob,dim=1)

        local_unlearn_score = model(unlearn_sample)
        local_unlearn_prob = local_unlearn_score.clone()
        local_prob = torch.cat((local_negative_prob,local_unlearn_prob),dim=1)
        local_prob = F.softmax(local_prob,dim=1)
        loss_KD = nn.KLDivLoss(reduction="batchmean")
        distill_loss = args.dist_mu*loss_KD(global_prob,local_prob.detach())

        unlearn_score = F.logsigmoid(-global_unlearn_score).squeeze(dim=1)

        if(args.uni_weight):
            unlearn_sample_loss = -unlearn_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            unlearn_sample_loss = -(subsampling_weight*unlearn_score).sum()/subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight*negative_score).sum()/subsampling_weight.sum()

        loss = (10*unlearn_sample_loss+negative_sample_loss)/2
        loss += distance_loss
        loss += distill_loss
        loss.backward()
        optimizer.step()
        log = {
            "distance_loss":distance_loss.item(),
            "distill_loss":distill_loss.item(),
            "unlearn_sample_loss": unlearn_sample_loss.item(),
            "negative_sample_loss": negative_sample_loss.item(),
            "loss": loss.item()
        }
        return log
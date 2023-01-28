import os
import numpy as np
import random
from sklearn.cluster import SpectralClustering

def clustersplit(in_path,out_dir,K=3):
    e2r = dict()
    rset = set()
    alltriples = []
    with open(in_path,"r",encoding="utf-8") as fin:
        for line in fin.readlines():
            h,r,t = line.strip().split()
            alltriples.append((h,r,t))
            rset.add(r)
            if h not in e2r:
                e2r[h] = dict()
            if t not in e2r:
                e2r[t] = dict()
            if r not in e2r[h]:
                e2r[h][r] = 1
            else:
                e2r[h][r] += 1
            if r not in e2r[t]:
                e2r[t][r] = 1
            else:
                e2r[t][r] += 1

    rlist = list(rset)
    rnumber = len(rlist)
    r2id = dict()
    for i in range(rnumber):
        r2id[rlist[i]] = i
    comatrix = [[0]*rnumber for _ in range(rnumber)]
    for e in e2r:
        corelations = list(e2r[e].keys())
        cornumber = len(corelations)
        for i in range(cornumber):
            for j in range(i+1,cornumber):
                comatrix[r2id[corelations[i]]][r2id[corelations[j]]] += 1
                comatrix[r2id[corelations[j]]][r2id[corelations[i]]] += 1

    sc = SpectralClustering(K,affinity="precomputed")
    labels = sc.fit_predict(comatrix)
    for label in range(K):
        print("cluster %i with relations %i" %(label,np.sum(labels==label)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outfiles = []
    for i in range(K):
        file_name = str(i)+".txt"
        out_path = os.path.join(out_dir,file_name)
        outfiles.append(open(out_path,"w",encoding="utf-8"))
    for (h,r,t) in alltriples:
        clientnum = labels[r2id[r]]
        outfiles[clientnum].write(h+"\t"+r+"\t"+t+"\n")
data_dir = "../Data/FB15k-237"
file_name = "all.txt"
file_path = os.path.join(data_dir,file_name)
out_dir = "../Data/FB15k-237/C3"
clustersplit(file_path,out_dir,K=3)
out_dir = "../Data/FB15k-237/C5"
clustersplit(file_path,out_dir,K=5)
out_dir = "../Data/FB15k-237/C10"
clustersplit(file_path,out_dir,K=10)
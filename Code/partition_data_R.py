import os
import random
def relationrandom(inpath,outdir,K=3):
    relations = []
    r2t = dict()
    with open(inpath,"r",encoding="utf-8")as fin:
        for line in fin.readlines():
            h,r,t = line.strip().split()
            if r not in relations:
                relations.append(r)
            if r not in r2t:
                r2t[r] = []
            r2t[r].append((h,r,t))
    relationnum = len(relations)
    offset = int(relationnum/K)
    print(relationnum,offset)
    random.shuffle(relations)
    relationlists = []
    left = 0
    right = offset
    for i in range(K):
        if i==K-1:
            right = relationnum
        relationlists.append(relations[left:right])
        left = right
        right += offset
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(K):
        outpath = os.path.join(outdir,str(i)+".txt")
        with open(outpath,"w",encoding="utf-8")as fout:
            for relation in relationlists[i]:
                for (h,r,t) in r2t[relation]:
                    fout.write(h+"\t"+r+"\t"+t+"\n")
data_dir = "../Data/FB15k-237"
inpath = os.path.join(data_dir,"all.txt")
out_dir = os.path.join(data_dir,"R3")
relationrandom(inpath,out_dir,K=3)
out_dir = os.path.join(data_dir,"R5")
relationrandom(inpath,out_dir,K=5)
out_dir = os.path.join(data_dir,"R10")
relationrandom(inpath,out_dir,K=10)
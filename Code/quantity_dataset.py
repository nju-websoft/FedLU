import os
import numpy as np
import networkx as nx
from scipy import stats
def quantify_graph(triples):
    entities = set()
    relations = set()
    edges = []
    indegree = dict()
    outdegree = dict()
    for (h,r,t) in triples:
        edges.append((h,t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
        if h not in outdegree:
            outdegree[h] = 1
        else:
            outdegree[h] += 1
        if t not in indegree:
            indegree[t] = 1
        else:
            indegree[t] += 1
    indegreelist = [(indegree[e] if e in indegree else 0) for e in entities]
    outdegreelist = [(outdegree[e] if e in outdegree else 0) for e in entities]
    indegreearray = np.array(indegreelist)
    outdegreearray = np.array(outdegreelist)
    indegreemu = np.mean(indegreearray)
    outdegreemu = np.mean(outdegreearray)
    indegreesigma = np.std(indegreearray)
    outdegreesigma = np.std(outdegreearray)
    indegreeskew = stats.skew(indegreearray)
    outdegreeskew = stats.skew(outdegreearray)
    indegreekurtosis = stats.kurtosis(indegreearray)
    outdegreekurtosis = stats.kurtosis(outdegreearray)
    print("general statistics:")
    print("num of entities/relations: %i,%i"%(len(entities),len(relations)))
    print("num of triples: %i"%(len(triples)))
    print("degree statistics:")
    print("mu of in/out degree: %f,%f"%(indegreemu,outdegreemu))
    print("std of in/out degree: %f,%f"%(indegreesigma,outdegreesigma))
    print("skew of in/out degree: %f,%f"%(indegreeskew,outdegreeskew))
    print("kurtosis of in/out degree: %f,%f"%(indegreekurtosis,outdegreekurtosis))
    G=nx.DiGraph(edges)
    numstronglyconnectedcomponents = nx.number_strongly_connected_components(G)
    numweaklyconnectedcomponents = nx.number_weakly_connected_components(G)
    avgclustercoefficient = nx.average_clustering(G)
    print("graph statistics:")
    print("number of strongly connected components: %i"%(numstronglyconnectedcomponents))
    print("number of weakly connected components: %i"%(numweaklyconnectedcomponents))
    print("average clustering coefficient: %f"%(avgclustercoefficient))

def read_triples(inpath):
    triples = []
    with open(inpath,"r",encoding="utf-8")as fin:
        for line in fin.readlines():
            h,r,t = line.strip().split()
            triples.append((h,r,t))
    return triples

def quantify_dataset(indir):
    if not os.path.exists(indir):
        return
    print("dir name: %s"%(indir))
    files = os.listdir(indir)
    for file in files:
        print("%s\n"%("_"*50))
        print("file name: %s"%(file))
        inpath = os.path.join(indir,file)
        triples = read_triples(inpath)
        quantify_graph(triples)

quantify_dataset("../Data/FB15k-237/R3")
quantify_dataset("../Data/FB15k-237/C3")
quantify_dataset("../Data/FB15k-237/R5")
quantify_dataset("../Data/FB15k-237/C5")
quantify_dataset("../Data/FB15k-237/R10")
quantify_dataset("../Data/FB15k-237/C10")
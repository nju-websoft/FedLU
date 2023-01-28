import os
def read_triples(in_path):
    triples = []
    with open(in_path,"r",encoding="utf-8") as fin:
        for line in fin.readlines():
            h,r,t = line.strip().split()
            triples.append((h,r,t))
    return triples
def read_dict(in_path):
    urls = set()
    with open(in_path,"r",encoding="utf-8") as fin:
        for line in fin.readlines():
            _,label = line.strip().split()
            urls.add(label)
    return urls
def write_triples(triples,out_path):
    with open(out_path,"w",encoding="utf-8") as fout:
        for (h,r,t) in triples:
            fout.write("%s\t%s\t%s\n"%(h,r,t))
def write_dict(urls,out_path):
    with open(out_path,"w",encoding="utf-8") as fout:
        id = 0
        for url in urls:
            fout.write("%i\t%s\n"%(id,url))
            id += 1
def write_set(urls,out_path):
    with open(out_path,"w",encoding="utf-8") as fout:
        for url in urls:
            fout.write("%s\n"%(url))
def myfunc(in_dir,out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trainset = []
    validset = []
    testsets = dict()
    unlearnsets = dict()
    entities = set()
    relations = set()
    for client_dir in os.listdir(in_dir):
        trainset.extend(read_triples(os.path.join(in_dir,client_dir,"train.txt")))
        validset.extend(read_triples(os.path.join(in_dir,client_dir,"valid.txt")))
        testsets[int(client_dir)]=read_triples(os.path.join(in_dir,client_dir,"test.txt"))
        unlearnsets[int(client_dir)]=read_triples(os.path.join(in_dir,client_dir,"unlearn.txt"))
        client_entities = read_dict(os.path.join(in_dir,client_dir,"entities.dict"))
        write_set(client_entities,os.path.join(out_dir,"e"+str(client_dir)+".set"))
        entities |= client_entities
        relations |= read_dict(os.path.join(in_dir,client_dir,"relations.dict"))
    print("train size: %i"%(len(trainset)))
    print("valid size: %i"%(len(validset)))
    print("entities size: %i"%(len(entities)))
    print("relations size: %i"%(len(relations)))
    write_triples(trainset,os.path.join(out_dir,"train.txt"))
    write_triples(validset,os.path.join(out_dir,"valid.txt"))
    for i in range(0,len(testsets)):
        testset = testsets[i]
        out_path = os.path.join(out_dir,"test"+str(i)+".txt")
        write_triples(testset,out_path)
    for i in range(0,len(unlearnsets)):
        unlearnset = unlearnsets[i]
        out_path = os.path.join(out_dir,"unlearn"+str(i)+".txt")
        write_triples(unlearnset,out_path)
    write_dict(entities,os.path.join(out_dir,"entities.dict"))
    write_dict(relations,os.path.join(out_dir,"relations.dict"))
    
myfunc("../Data/FB15k-237/C3FL","../Data/FB15k-237/C3GlobalUnlearn")
myfunc("../Data/FB15k-237/C5FL","../Data/FB15k-237/C5GlobalUnlearn")
myfunc("../Data/FB15k-237/C10FL","../Data/FB15k-237/C10GlobalUnlearn")

myfunc("../Data/FB15k-237/R3FL","../Data/FB15k-237/R3GlobalUnlearn")
myfunc("../Data/FB15k-237/R5FL","../Data/FB15k-237/R5GlobalUnlearn")
myfunc("../Data/FB15k-237/R10FL","../Data/FB15k-237/R10GlobalUnlearn")
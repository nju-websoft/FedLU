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
def construct_globaldataset(in_dir,out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trainset = []
    validset = []
    testsets = []
    entities = set()
    relations = set()
    for client_dir in os.listdir(in_dir):
        trainset.extend(read_triples(os.path.join(in_dir,client_dir,"train.txt")))
        validset.extend(read_triples(os.path.join(in_dir,client_dir,"valid.txt")))
        testsets.append(read_triples(os.path.join(in_dir,client_dir,"test.txt")))
        client_entities = read_dict(os.path.join(in_dir,client_dir,"entities.dict"))
        write_set(client_entities,os.path.join(out_dir,"e"+str(client_dir)+".set"))
        entities |= client_entities
        relations |= read_dict(os.path.join(in_dir,client_dir,"relations.dict"))
    print("train size: %i"%(len(trainset)))
    print("valid size: %i"%(len(validset)))
    print("test size: %i"%(sum([len(testset) for testset in testsets])))
    print("entities size: %i"%(len(entities)))
    print("relations size: %i"%(len(relations)))
    write_triples(trainset,os.path.join(out_dir,"train.txt"))
    write_triples(validset,os.path.join(out_dir,"valid.txt"))
    for i in range(len(testsets)):
        testset = testsets[i]
        out_path = os.path.join(out_dir,"test"+str(i)+".txt")
        write_triples(testset,out_path)
    write_dict(entities,os.path.join(out_dir,"entities.dict"))
    write_dict(relations,os.path.join(out_dir,"relations.dict"))

construct_globaldataset("../Data/FB15k-237/R3FL","../Data/FB15k-237/R3Global")
construct_globaldataset("../Data/FB15k-237/C3FL","../Data/FB15k-237/C3Global")
construct_globaldataset("../Data/FB15k-237/R5FL","../Data/FB15k-237/R5Global")
construct_globaldataset("../Data/FB15k-237/C5FL","../Data/FB15k-237/C5Global")
construct_globaldataset("../Data/FB15k-237/R10FL","../Data/FB15k-237/R10Global")
construct_globaldataset("../Data/FB15k-237/C10FL","../Data/FB15k-237/C10Global")
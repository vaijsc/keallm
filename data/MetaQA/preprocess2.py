import re
import json
import pandas as pd
import os
from tqdm import tqdm
# h_rt = {}
# t_rh = {}
# with open("kb.txt") as f:
#     for l in f.readlines():
#         h,r,t = l.strip().split("|")
#         if h not in h_rt:
#             h_rt[h] = [[r, t]]
#         else:
#             h_rt[h].append([r, t])

#         if t not in t_rh:
#             t_rh[t] = [[r, h]]
#         else:
#             t_rh[t].append([r, h])

# for mode in ["train", "dev", "test"]:
#     for k in range(1,4):
#         new_samples = []
#         with open(f"{k}-hop/{mode}.txt") as f:
#             for l in f.readlines():
#                 q, a = l.strip().split("\t")
#                 a = a.split("|")
#                 try:
#                     q_entity = re.findall(r"(?<=\[)[^\[\]]*(?=\])", q)[0]
#                 except:
#                     import IPython; IPython.embed(); exit(1)
#                 assert q_entity in h_rt or q_entity in t_rh
#                 if q_entity in h_rt:
#                     for r, t in h_rt[q_entity]:
#                         if t in a:
#                             new_samples.append(dict(
#                                             question=q, 
#                                             triples=[dict(h=q_entity,r=r,t=aa) for aa in a]))
#                             break
#                 else:
#                     # change the direction
#                     for r, t in t_rh[q_entity]:
#                         if t in a:
#                             new_samples.append(dict(
#                                             question=q, 
#                                             triples=[dict(h=q_entity,r=r+" [inverse]",t=aa) for aa in a]))
#                             break
#         with open(f"{k}-hop/{mode}.json", "w") as f:
#             for s in new_samples:
#                 f.write(json.dumps(s, ensure_ascii=False) + '\n')
def step_one():
    # dataset = load_dataset("rmanluo/RoG-webqsp")
    # dataset = load_from_disk("./MetaQA")
    # dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    with open("./kb.txt") as f:
        dataset = f.read().split("\n")[:-1]
    with open("./entity2text.txt") as f:
        entity2text = f.read().split("\n")[:-1]
    for idx in range(len(entity2text)):
       
        entity2text[idx] = entity2text[idx].split("\t")[1]
    
    with open("./relation2text.txt") as f:
        relation2text = f.read().split("\n")[:-1]
    for idx in range(len(relation2text)):
      
        relation2text[idx] = relation2text[idx].split("\t")[1]

    sample = []
    for i in tqdm(range(len(dataset))):
        triplet = dataset[i].split('|')
        h, r, t = triplet
        h = entity2text.index(h)
        r = relation2text.index(r.replace("_", " "))
        t = entity2text.index(t)
        sp = f"{h}\t{r}\t{t}"
        sample.append(sp)
    import random
    random.shuffle(sample)
    with open(f"train.tsv", "w") as f:
        f.write("\n".join(sample[:129000]) + '\n')
    with open(f"dev.tsv", "w") as f:
        f.write("\n".join(sample[129000:130000]) + '\n')
    with open(f"test.tsv", "w") as f:
        f.write("\n".join(sample[130000:]) + '\n')

def step_one_further():
    # dataset = load_dataset("rmanluo/RoG-webqsp")
    # dataset = load_from_disk("./MetaQA")
    # dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    # with open("./kb.txt") as f:
    #     dataset = f.read().split("\n")[:-1]
    with open("./entity2text.txt") as f:
        entity2text = f.read().split("\n")[:-1]
    for idx in range(len(entity2text)):
       
        entity2text[idx] = entity2text[idx].split("\t")[1]
    
    with open("./relation2text.txt") as f:
        relation2text = f.read().split("\n")[:-1]
    for idx in range(len(relation2text)):
      
        relation2text[idx] = relation2text[idx].split("\t")[1]
    
    dataset = pd.read_json("./1-hop/test.json", lines=True)
    
    sample = []
    for i in tqdm(range(len(dataset))):
        triplets = dataset['triples'][i]
        for triplet in triplets:
            
            h, r, t = triplet['h'], triplet['r'], triplet['t']
            if '[inverse]' in r:
                r = r.replace(" [inverse]", "")
                tmp = h
                h = t
                t = tmp
            h = entity2text.index(h)
            r = relation2text.index(r.replace("_", " "))
            t = entity2text.index(t)
            sp = f"{h}\t{r}\t{t}"
            sample.append(sp)
    # import random
    # random.shuffle(sample)
    # with open(f"train.tsv", "w") as f:
    #     f.write("\n".join(sample[:129000]) + '\n')
    # with open(f"dev.tsv", "w") as f:
    #     f.write("\n".join(sample[129000:130000]) + '\n')
    with open(f"test.tsv", "w") as f:
        f.write("\n".join(sample) + '\n')

if __name__ == "__main__":
    step_one_further()

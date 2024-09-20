import re
import json
h_rt = {}
t_rh = {}
with open("kb.txt") as f:
    for l in f.readlines():
        h,r,t = l.strip().split("|")
        if h not in h_rt:
            h_rt[h] = [[r, t]]
        else:
            h_rt[h].append([r, t])

        if t not in t_rh:
            t_rh[t] = [[r, h]]
        else:
            t_rh[t].append([r, h])

for mode in ["train", "dev", "test"]:
    for k in range(1,4):
        new_samples = []
        with open(f"{k}-hop/{mode}.txt") as f:
            for l in f.readlines():
                q, a = l.strip().split("\t")
                a = a.split("|")
                try:
                    q_entity = re.findall(r"(?<=\[)[^\[\]]*(?=\])", q)[0]
                except:
                    import IPython; IPython.embed(); exit(1)
                assert q_entity in h_rt or q_entity in t_rh
                if q_entity in h_rt:
                    for r, t in h_rt[q_entity]:
                        if t in a:
                            new_samples.append(dict(
                                            question=q, 
                                            triples=[dict(h=q_entity,r=r,t=aa) for aa in a]))
                            break
                else:
                    # change the direction
                    for r, t in t_rh[q_entity]:
                        if t in a:
                            new_samples.append(dict(
                                            question=q, 
                                            triples=[dict(h=q_entity,r=r+" [inverse]",t=aa) for aa in a]))
                            break
        with open(f"{k}-hop/{mode}.json", "w") as f:
            for s in new_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

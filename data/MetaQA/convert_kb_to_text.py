kg = []
with open("kb.txt") as f:
    for l in f.readlines():
        h,r,t = l.strip().split("|")
        kg.append([h,r,t])

with open("kg.txt", 'w') as f:
    for h,r,t in kg:
        f.write("\t".join([h,r,t]) + '\n')
        f.write("\t".join([h,r + " [inverse]",t]) + '\n')
    
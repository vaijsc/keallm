relation2text = {}
cnt = 0
with open("kb.txt") as f:
    for l in f.readlines():
        h,r,t = l.strip().split("|")
        if r not in relation2text:
            relation2text[r] = [str(cnt), r.replace("_"," ")]
            cnt += 1

with open('relation2text.txt', 'w') as f:
    for r, text in relation2text.values():
        f.write("\t".join([r,text]) + '\n')
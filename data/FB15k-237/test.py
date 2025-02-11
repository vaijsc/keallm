from distutils.log import debug
import os

from collections import Counter

cnt = {}
with open("train.tsv") as file:
    for line in file.readlines():
        h,r,t = line.strip().split('\t')
        if r not in cnt:
            cnt[r] = 1
        else:
            cnt[r] += 1



import IPython; IPython.embed(); exit(1)



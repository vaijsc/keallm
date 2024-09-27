# total_relations = []
# with open("relation2text_origin.txt", "r") as file:
#     for line in file.readlines():
#         rel = line.strip().split("\t")[0].split("/")[-1]
#         rel_id = line.strip().split("\t")[0]
#         total_relations.append("\t".join([rel_id, rel.replace("_", " ")]))

# with open("relation2text.txt", "w") as file:
#     for l in total_relations:
#         file.write(l + "\n")



from collections import Counter
total_types = Counter()

with open("./entity2type.txt", 'r') as file:
    for line in file.readlines():
        types = line.strip().split("\t")[1:]
        for t in types:
            total_types

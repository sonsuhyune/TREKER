import json

train_data = json.load(open("/home/data/ssh5131/code/naver/EM_SAIS/dataset/original_TREK/train_annotated.json"))
dev_data = json.load(open("/home/data/ssh5131/code/naver/EM_SAIS/dataset/original_TREK/dev.json"))
test_data = json.load(open("/home/data/ssh5131/code/naver/EM_SAIS/dataset/original_TREK/test.json"))

label_dict =  {"P1": "org:dissolved", "P2": "org:founded", "P3": "org:place_of_headquarters", "P4": "org:alternate_names", "P5": "org:member_of", "P6": "org:members", "P7": "org:political/religious_affiliation", "P8": "org:product", "P9": "org:founded_by", "P10": "org:top_members/employees", "P11": "per:place_of_birth", "P12": "per:place_of_death", "P13": "per:place_of_residence", "P14": "per:origin", "P15": "per:employee_of", "P16": "per:schools_attended", "P17": "per:alternate_names", "P18": "per:parents", "P19": "per:children", "P20": "per:siblings", "P21": "per:spouse", "P22": "per:other_family", "P23": "per:colleagues", "P24": "per:product", "P25": "per:religion", "P26": "per:title"}

# lable_dict_={v: k for k, v in label_dict.items()}
# for d in data:
#     for l in d["labels"]:
#         if l["r"] == '':
#             l["r"] = lable_dict_[l["r_name"]]

count=0
total=0
for doc in train_data:
    for label in doc['labels']:
        total+=1
        if len(doc["vertexSet"]) <= label["h"] or len(doc["vertexSet"]) <= label["t"] :
            count+=1

        # breakpoint()

print(f"train: {count}/{total}")
count=0
total=0
for doc in dev_data:
    for label in doc['labels']:
        total+=1
        if len(doc["vertexSet"]) <= label["h"] or len(doc["vertexSet"]) <= label["t"]:
            count += 1

        # breakpoint()
print(f"dev: {count}/{total}")
count=0
total=0
for doc in test_data:
    for label in doc['labels']:
        total+=1
        if len(doc["vertexSet"]) <= label["h"] or len(doc["vertexSet"]) <= label["t"]:
            count += 1
        # breakpoint()
print(f"test: {count}/{total}")

# for d in data:
#     for l in d["labels"]:
#         if l["r"] == '':
#             print(l)
#
# with open(f"/home/data/ssh5131/code/naver/EM_SAIS/dataset/TREK/test.json", 'w', encoding='utf-8') as write_file:
#     json.dump(data, write_file, indent='\t',ensure_ascii=False)
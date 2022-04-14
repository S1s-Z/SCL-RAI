from datasets import load_dataset
import json
from operator import itemgetter

tag2index_conll = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
index2tag_conll = dict(zip(tag2index_conll.values(), tag2index_conll.keys()))
#print(index2tag_conll)

datasets = load_dataset("conll2003")
#print(datasets['train'][0])
json_answer_list = []

for data in datasets['test']:
    temp_dict = {}
    if data['tokens'] != []:
        #temp_dict["sentence"] = str(data['tokens']).replace("\"","'").replace(r"\n","")#双引号换单引号
        temp_dict["sentence"] = data['tokens']
        temp_dict["tags"] = data['ner_tags']
        json_answer_list.append(temp_dict)

final_list = []
for data in json_answer_list[:]:
    temp_dict = {}
    labeled_entities = []
    for i in [1,3,5,7]:#针对每个B标签进行循环
        if i in data['tags']:
            #print(i) #输出轮到的tag
            index_num = [x for x, y in list(enumerate(data['tags'])) if y == i] #计算所有tag的所有位置
            for j in index_num:
                if j+1 == len(data['tags']):#如果出现在最后一位
                    labeled_entities.append((j, j, index2tag_conll[i][2:]))
                else:#如果不在最后一位
                    if data['tags'][j+1] == i+1:#如果下一位等于其对应的I,则继续延伸计算出span的长度
                        temp_flag = 0
                        while data['tags'][j+temp_flag+1] == i+1:
                            temp_flag = temp_flag + 1
                            if j+temp_flag >= len(data['tags'])-1:
                                break
                        labeled_entities.append((j, j+temp_flag, index2tag_conll[i][2:]))
                    else:#如果下一位不是其对应的I，则其span长度为1
                        labeled_entities.append((j, j, index2tag_conll[i][2:]))

    #print(data['tags'])
    #print(sorted(labeled_entities, key=itemgetter(0)))
    #print(data['sentence'])
    temp_dict['sentence'] = str(data['sentence'])
    temp_dict['labeled entities'] = str(sorted(labeled_entities, key=itemgetter(0)))
    final_list.append(temp_dict)

jsObj = json.dumps(final_list, indent=1)
fileObject = open('test.json', 'w')
fileObject.write(jsObj)











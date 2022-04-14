import json
from operator import itemgetter

#tag2index_news = {'O': 0, 'B-cp': 1, 'I-cp': 2, 'B-pp': 3,'I-pp': 4,'B-gg': 5,'I-gg': 6,'B-xh': 7,'I-xh': 8,'B-yl': 9,'I-yl': 10}
tag2index_news = {'O': 0, 'B-PER': 1, 'I-PER': 2}
index2tag_news = dict(zip(tag2index_news.values(), tag2index_news.keys()))
print(index2tag_news)

f = open("unlabel_data/MSRA/ds_fa.txt")
line = f.readlines()
temp_list = []
temp_all_list = []
temp_answer_all_list = []
temp_answer_list = []
for i in line:
    if i.split()!=[]:
        temp_list.append(i.split('\t')[0])
        temp_answer_list.append(tag2index_news[i.split('\t')[1].replace("\n", "")])
        #print(temp_answer_list)
    else:
        temp_all_list.append(temp_list)
        temp_answer_all_list.append(temp_answer_list)
        temp_list = []
        temp_answer_list = []
#print(temp_all_list )
#print(temp_answer_all_list)
f.close()


json_answer_list = []
for i in range(len(temp_answer_all_list)):
    #print(temp_all_list[i])
    temp_dict = {}
    temp_dict["sentence"] = temp_all_list[i]
    temp_dict["tags"] = temp_answer_all_list[i]
    json_answer_list.append(temp_dict)
print(json_answer_list)

final_list = []
for data in json_answer_list[:]:
    temp_dict = {}
    labeled_entities = []
    for i in [1]:#针对每个B标签进行循环
        if i in data['tags']:
            #print(i) #输出轮到的tag
            index_num = [x for x, y in list(enumerate(data['tags'])) if y == i] #计算所有tag的所有位置
            for j in index_num:
                if j+1 == len(data['tags']):#如果出现在最后一位
                    labeled_entities.append((j, j, index2tag_news[i][2:]))
                else:#如果不在最后一位
                    if data['tags'][j+1] == i+1:#如果下一位等于其对应的I,则继续延伸计算出span的长度
                        temp_flag = 0
                        while data['tags'][j+temp_flag+1] == i+1:
                            temp_flag = temp_flag + 1
                            if j+temp_flag >= len(data['tags'])-1:
                                break
                        labeled_entities.append((j, j+temp_flag, index2tag_news[i][2:]))
                    else:#如果下一位不是其对应的I，则其span长度为1
                        labeled_entities.append((j, j, index2tag_news[i][2:]))

    #print(data['tags'])
    #print(sorted(labeled_entities, key=itemgetter(0)))
    #print(data['sentence'])
    temp_dict['sentence'] = str(data['sentence'])
    temp_dict['labeled entities'] = str(sorted(labeled_entities, key=itemgetter(0)))
    final_list.append(temp_dict)
print(final_list)
jsObj = json.dumps(final_list, indent=1,ensure_ascii=False)
fileObject = open('ds_fa.json', 'w')
fileObject.write(jsObj)
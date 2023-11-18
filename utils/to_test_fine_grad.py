import json
import re
#re匹配模式
pattern1 = r'(？|\?|!|。|！)'
pattern2 = r"\n|\t|\r"
pattern3 = r"？|\?|!|。|！"
max_sentence = 7
#处理获得的一列数据text，得到不同实体匹配输入到关系抽取模型中
def RE_text_process(text,entities):
    ent1_list,ent2_list,text_list,ent1_id,ent2_id, = [],[],[],[],[]
    splited_text = re.split(pattern3,text)
    for i in range(len(entities)):
        for j in range(i+1,len(entities)):
            ent1 = [text[entities[i]["start_offset"]:entities[i]["end_offset"]],entities[i]["label"],entities[i]["id"],entities[i]["start_offset"],entities[i]["end_offset"]]
            ent2 = [text[entities[j]["start_offset"]:entities[j]["end_offset"]],entities[j]["label"],entities[j]["id"],entities[j]["start_offset"],entities[j]["end_offset"]]
            extract_text,flag=text_split(splited_text,ent1,ent2)
            if flag<=max_sentence:
                ent1_list.append(ent1[0])
                ent2_list.append(ent2[0])
                ent1_id.append(ent1[2])
                ent2_id.append(ent2[2])
                text_list.append(extract_text)
    return ent1_list,ent2_list,text_list,ent1_id,ent2_id

def text_split(text,ent1,ent2):
    ent_location = sorted([ent1[3],ent1[4],ent2[3],ent2[4]])
    min,max = ent_location[0],ent_location[-1]
    accumulate = 0
    out_text = ""
    sentence_cum = 0
    for t in text:
        last = accumulate
        accumulate+= len(t)
        if (min>last and min<accumulate)or(min<last and max>accumulate)or(max>last and max<accumulate):
            sentence_cum+=1
            out_text+=re.sub(pattern2,"",t)
    return out_text,sentence_cum


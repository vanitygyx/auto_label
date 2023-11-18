import json
import re

#处理获得的一列数据text
def NER_text_process(line):
    pattern = r'(？|\?|!|。|！)'
    text = line
    sub_splited_text = re.sub(r"[\s,\r]","",text)
    sub_splited_text = re.sub(r"[􀦋,􀤋]","",sub_splited_text)
    #print(splited_text)
    sub_splited_text = re.split(pattern,sub_splited_text)
    splited_text = re.split(pattern,text)
    return sub_splited_text,splited_text

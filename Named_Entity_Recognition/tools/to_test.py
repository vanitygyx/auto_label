import json
import re

#处理获得的一列数据text
def NER_text_process(line):
    pattern = r'？|\?|!|。|！'
    text = line['text']
    splited_text = re.sub(r"\n|\t|\r","",text)
    #print(splited_text)
    splited_text = re.split(pattern,text)
    return splited_text

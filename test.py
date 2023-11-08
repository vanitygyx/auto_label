from flask import Flask, request, jsonify
from tools.to_test import NER_text_process
from run_ner_crf_new import ner_test
from utils.to_test_fine_grad import RE_text_process
from utils.auto_extraction import re_test
import json
jsonl_file = "all.jsonl"

if __name__ == '__main__':      
    with open(jsonl_file,'r',encoding="utf-8") as f:
        d = f.readline()
        line = json.loads(d)
        split_text= NER_text_process(line)
        entities = ner_test(split_text)
        print(entities)
